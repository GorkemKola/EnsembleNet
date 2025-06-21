import zipfile
from ensemblenet.utils import logger
from ensemblenet.entity import DataIngestionConfig
import shutil
import subprocess
import random

class DataIngestion:
    def __init__(
            self,
            config: DataIngestionConfig
        ) -> None:
        self.config = config

    # ... (download_file method remains the same) ...
    def download_file(self):
        '''
        Description:
            Downloads the dataset from Kaggle using the Kaggle API.
            Returns a status string.
        '''
        try:
            dataset_id = self.config.source_kaggle_dataset_id
            download_dir = self.config.root_dir # Directory to download the zip file into
            local_data_file = self.config.local_data_file

            # Check if the zip file exists first
            if local_data_file.exists():
                logger.info(f"Zip file '{local_data_file}' already exists.")
                # More robust check: if split dirs are populated, assume everything is done
                if self.config.train_dir.exists() and any(f for f in self.config.train_dir.glob('*/*') if f.is_file()): # Check for files within class dirs
                     logger.info("Split directories seem populated. Skipping download, extraction, and split.")
                     return "Skipped - Already Processed"
                else:
                    logger.info("Zip file exists, but split directories seem empty or incomplete. Proceeding.")
                    # No return, continue to check extraction/split
            else:
                logger.info(f"Downloading dataset '{dataset_id}' from Kaggle into '{download_dir}'...")
                command = [
                    "kaggle", "datasets", "download",
                    "-d", dataset_id,
                    "-p", str(download_dir),
                ]
                result = subprocess.run(command, capture_output=True, text=True, check=False)

                if result.returncode == 0:
                    expected_zip_filename = dataset_id.split('/')[-1] + ".zip"
                    potential_downloaded_path = download_dir / expected_zip_filename
                    actual_local_data_file = self.config.local_data_file

                    if not actual_local_data_file.exists():
                        if potential_downloaded_path.exists():
                            logger.warning(f"Downloaded file seems to be '{potential_downloaded_path}', but config expects '{actual_local_data_file}'. Renaming.")
                            try:
                                potential_downloaded_path.rename(actual_local_data_file)
                            except OSError as e:
                                logger.error(f"Error renaming downloaded file: {e}. Trying to copy and remove.")
                                try:
                                     shutil.copy2(potential_downloaded_path, actual_local_data_file)
                                     potential_downloaded_path.unlink()
                                     logger.info("Successfully copied and removed original download.")
                                except Exception as copy_e:
                                     logger.error(f"Failed to copy/delete after rename error: {copy_e}")
                                     raise FileNotFoundError(f"Downloaded file '{potential_downloaded_path}' exists but couldn't be moved/copied to '{actual_local_data_file}'.")

                        else:
                            logger.error(f"Kaggle command succeeded, but expected file '{actual_local_data_file}' (or '{potential_downloaded_path}') not found.")
                            logger.error(f"Kaggle CLI stdout:\n{result.stdout}")
                            logger.error(f"Kaggle CLI stderr:\n{result.stderr}")
                            raise FileNotFoundError(f"Expected file '{actual_local_data_file}' not found after download attempt.")
                    logger.info(f"Successfully downloaded '{dataset_id}' to '{actual_local_data_file}'")
                else:
                    # ... (rest of the error handling) ...
                    logger.error(f"Failed to download dataset '{dataset_id}'.")
                    logger.error(f"Return Code: {result.returncode}")
                    logger.error(f"Kaggle CLI stdout:\n{result.stdout}")
                    logger.error(f"Kaggle CLI stderr:\n{result.stderr}")
                    if "401" in result.stderr or "authenticate" in result.stderr.lower():
                         logger.error("Authentication error: Ensure 'kaggle.json' is correctly placed and configured.")
                    elif "404" in result.stderr or "not found" in result.stderr.lower():
                         logger.error(f"Dataset '{dataset_id}' not found on Kaggle. Check the dataset ID.")
                    elif "429" in result.stderr or "Too Many Requests" in result.stderr:
                         logger.error("Rate limit exceeded. Please wait before trying again.")
                    raise Exception(f"Kaggle download failed with stderr: {result.stderr}")

        except FileNotFoundError as e:
             if "kaggle" in str(e):
                  logger.error("The 'kaggle' command was not found. Please ensure the Kaggle CLI is installed and in your system's PATH.")
                  raise RuntimeError("Kaggle CLI not found. Please install it (`pip install kaggle`) and configure it.") from e
             else:
                  logger.error(f"An unexpected FileNotFoundError occurred: {e}")
                  raise e # Re-raise other FileNotFoundError
        except Exception as e:
            logger.error(f"An error occurred during file download: {e}")
            raise e
        return "Download Complete"

    def extract_zip_file(self):
        '''
        Description:
            Extracts the downloaded zip file into the unzip directory.
            Returns a status string.
        '''
        unzip_path = self.config.unzip_dir
        local_data_file = self.config.local_data_file

        # Check if extraction seems complete AND split seems complete
        if self.config.train_dir.exists() and any(f for f in self.config.train_dir.glob('*/*') if f.is_file()): # Check for files within class dirs
             logger.info(f"Split directories seem populated. Assuming extraction is also complete. Skipping extraction.")
             return "Skipped - Already Extracted/Split"

        try:
            if not local_data_file.exists():
                 logger.error(f"Cannot extract. Zip file '{local_data_file}' does not exist. Run download first.")
                 raise FileNotFoundError(f"Zip file '{local_data_file}' not found for extraction.")

            logger.info(f"Extracting '{local_data_file}' into '{unzip_path}'...")
            unzip_path.mkdir(parents=True, exist_ok=True) # Ensure it exists

            with zipfile.ZipFile(local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info(f"Successfully extracted '{local_data_file}' to '{unzip_path}'")

            extracted_items = list(unzip_path.iterdir())
            if not extracted_items:
                logger.warning(f"Extraction finished, but the directory '{unzip_path}' is empty. Check the zip file content and structure.")
            else:
                logger.info(f"Extraction seems successful, found {len(extracted_items)} items in '{unzip_path}'.")

        except zipfile.BadZipFile:
             logger.error(f"Error: '{local_data_file}' is not a valid zip file or is corrupted.")
             raise
        except Exception as e:
             logger.error(f"An error occurred during zip extraction: {e}")
             raise e
        return "Extraction Complete"


    # ... (split_data method remains the same) ...
    def split_data(self, train_ratio=0.7, test_ratio=0.2): # Removed unused val_ratio param
        """
        Splits the data from unzip_dir into train, test, and validation sets.
        Returns a status string.
        """
        logger.info("Starting data splitting...")
        source_dir = self.config.unzip_dir
        train_dir = self.config.train_dir
        test_dir = self.config.test_dir
        val_dir = self.config.val_dir

        # Check if splitting appears complete
        if train_dir.exists() and any(f for f in train_dir.glob('*/*') if f.is_file()):
            logger.info("Train directory already exists and seems populated. Assuming split is already done. Skipping.")
            return "Skipped - Already Split"

        if not source_dir.exists() or not any(source_dir.iterdir()):
             logger.error(f"Source directory for splitting '{source_dir}' does not exist or is empty. Run extraction first.")
             if self.config.local_data_file.exists():
                 logger.error(f"Zip file '{self.config.local_data_file}' exists, but extraction directory is missing/empty. Try running extraction again.")
             raise FileNotFoundError(f"Source directory '{source_dir}' not found or empty.")

        # Check ratios
        if not (0 < train_ratio < 1 and 0 < test_ratio < 1 and train_ratio + test_ratio < 1):
             logger.error(f"Invalid ratios: train={train_ratio}, test={test_ratio}. They must be between 0 and 1, and sum to less than 1.")
             raise ValueError("Invalid train/test ratios for splitting.")
        effective_val_ratio = 1.0 - train_ratio - test_ratio
        logger.info(f"Using split ratios: Train={train_ratio:.2f}, Test={test_ratio:.2f}, Validation={effective_val_ratio:.2f}")

        # --- Start: Corrected Logic for finding data_root_in_zip ---
        data_root_in_zip = None
        # Explicitly check for the known intermediate directory name(s) first.
        # Handle potential case variations observed in logs.
        possible_intermediate_names = ["256_ObjectCategories", "256_objectcategories"]

        for name in possible_intermediate_names:
            potential_dir = source_dir / name
            if potential_dir.is_dir():
                # IMPORTANT: Check if this directory actually contains class-like subdirs
                # (e.g., directories starting with digits like '001.', '002.')
                # This prevents selecting an empty or incorrect intermediate folder.
                subdirs = [d for d in potential_dir.iterdir() if d.is_dir() and d.name[:3].isdigit()]
                if subdirs:
                    logger.info(f"Found valid intermediate directory '{name}' containing class subdirectories. Using it as data root.")
                    data_root_in_zip = potential_dir
                    break # Found the correct one, stop checking

        # If no valid intermediate directory was found, check if source_dir itself contains the classes
        if data_root_in_zip is None:
            logger.warning(f"Standard intermediate directory not found or not validated. Checking if '{source_dir.name}' contains class directories directly.")
            items_in_source = list(source_dir.iterdir())
            # Check for class-like dirs directly in source_dir
            class_like_dirs_in_source = [d for d in items_in_source if d.is_dir() and d.name[:3].isdigit()]

            # Use a threshold - Caltech256 has > 250 classes.
            if len(class_like_dirs_in_source) > 100: # Expecting many classes if this is the root
                 logger.info(f"Found {len(class_like_dirs_in_source)} class-like directories directly in '{source_dir.name}'. Assuming this is the data root.")
                 data_root_in_zip = source_dir
            else:
                # Log detailed info for debugging before failing
                logger.error(f"Cannot determine data root for splitting.")
                logger.error(f"Checked for intermediate dirs: {possible_intermediate_names} in {source_dir}")
                logger.error(f"Checked for direct class dirs (like '###.*') in {source_dir}")
                logger.error(f"Contents of '{source_dir}': {[item.name for item in source_dir.iterdir()]}")
                # Also log contents of potential intermediate dirs if they exist
                for name in possible_intermediate_names:
                     potential_dir = source_dir / name
                     if potential_dir.is_dir():
                         logger.error(f"Contents of potential intermediate '{name}': {[item.name for item in potential_dir.iterdir()]}")
                return "Split Failed - Cannot Find Data Root Structure"
        # --- End: Corrected Logic ---


        class_dirs = [d for d in data_root_in_zip.iterdir() if d.is_dir()]
        # Filter out potential non-class dirs like '.DS_Store' or others if necessary
        class_dirs = [d for d in class_dirs if not d.name.startswith('.')]
        # Optionally, be more strict:
        # class_dirs = [d for d in class_dirs if d.name[:3].isdigit()]

        if not class_dirs:
            logger.error(f"No valid class subdirectories (like '001.ak47') found in the determined data root '{data_root_in_zip}'. Cannot perform split.")
            return "Split Failed - No Class Dirs Found in Root"

        num_classes = len(class_dirs)
        logger.info(f"Found {num_classes} classes in '{data_root_in_zip}'. Starting file distribution.")

        # Ensure split directories exist (CM should have done this, but double-check)
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        total_files_copied = 0
        # --- Loop through classes and copy files (Logic from previous working version) ---
        for i, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            # Use debug for potentially verbose per-class logs
            logger.debug(f"Processing class {i+1}/{num_classes}: {class_name} from {class_dir}")

            # List image files within this specific class directory
            files = [f for f in class_dir.glob('*') if f.is_file() and not f.name.startswith('.') and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff']]

            if not files:
                logger.warning(f"  No suitable image files found in class directory: {class_dir}. Skipping this class.")
                continue # Skip to the next class

            random.shuffle(files)
            n_files = len(files)
            n_train = int(n_files * train_ratio)
            n_test = int(n_files * test_ratio)
            # n_val is the remainder
            n_val = n_files - n_train - n_test

            # Handle edge case: ensure train/test/val get at least one file if possible and n_files > 0
            if n_files > 0 and n_train == 0: n_train = 1
            if n_files > n_train and n_test == 0: n_test = 1
            # Recalculate n_val after potential adjustments
            n_val = n_files - n_train - n_test
            # Ensure n_val isn't negative if adjustments took needed files
            if n_val < 0:
                 # This implies n_train + n_test > n_files after adjustment. Prioritize train, then test.
                 if n_train + n_test > n_files: n_test = n_files - n_train # Adjust test down first
                 if n_test < 0 : n_test = 0 # Ensure test isn't negative
                 if n_train > n_files: n_train = n_files # Adjust train if needed (shouldn't happen with test adjust)
                 n_val = 0 # Val gets zero if adjustments used all files

            logger.debug(f"  Splitting {n_files} files: Train={n_train}, Test={n_test}, Val={n_val}")

            train_files = files[:n_train]
            test_files = files[n_train : n_train + n_test]
            val_files = files[n_train + n_test :] # Takes the rest

            # Create destination class directories
            (train_dir / class_name).mkdir(parents=True, exist_ok=True)
            (test_dir / class_name).mkdir(parents=True, exist_ok=True)
            (val_dir / class_name).mkdir(parents=True, exist_ok=True)

            # Copy files
            files_copied_this_class = 0
            try:
                for f_list, dest_dir in [(train_files, train_dir), (test_files, test_dir), (val_files, val_dir)]:
                    dest_class_dir = dest_dir / class_name
                    for f in f_list:
                        try:
                            shutil.copy2(str(f), str(dest_class_dir / f.name)) # copy2 preserves metadata
                            files_copied_this_class += 1
                        except Exception as file_copy_error:
                             logger.warning(f"    Could not copy file {f.name} to {dest_class_dir}: {file_copy_error}")
                             # Continue with other files in the same class/split

            except Exception as copy_error:
                 logger.error(f"  Major error during file copying for class {class_name}: {copy_error}")
                 # Depending on severity, you might want to stop or continue
                 # raise copy_error # Option: Stop execution
                 continue # Option: Continue with the next class

            total_files_copied += files_copied_this_class
            logger.debug(f"  Finished class {class_name}, copied {files_copied_this_class} files.")


        if total_files_copied == 0 and num_classes > 0:
             logger.error("Data splitting loop completed, but 0 files were copied. Check file permissions, disk space, or image file extensions.")
             return "Split Failed - No Files Copied"
        elif num_classes == 0:
             # This case is handled earlier, but double-check
             logger.error("Data splitting failed because no class directories were processed.")
             return "Split Failed - No Classes Processed"
        else:
            logger.info(f"Data splitting completed. Copied {total_files_copied} files across {num_classes} classes.")
            return "Split Complete"


    def cleanup_unzip_dir(self):
        """
        Removes the temporary directory where files were initially unzipped,
        if the configuration flag `cleanup_unzip_dir_after_split` is True.
        """
        if not self.config.cleanup_unzip_dir_after_split:
            logger.info("Cleanup of unzip directory is disabled in the configuration. Skipping removal.")
            return # Exit the function if cleanup is disabled

        unzip_path = self.config.unzip_dir
        if unzip_path.exists() and unzip_path.is_dir():
            logger.info(f"Attempting to remove temporary extraction directory: {unzip_path}")
            try:
                shutil.rmtree(unzip_path)
                logger.info(f"Successfully removed directory: {unzip_path}")
            except OSError as e:
                # OSError is common for permission errors or if dir is in use
                logger.error(f"Error removing directory {unzip_path}: {e}. Check permissions or if it's in use.")
            except Exception as e:
                logger.error(f"An unexpected error occurred while removing directory {unzip_path}: {e}")
        else:
            logger.warning(f"Directory {unzip_path} not found or is not a directory. Skipping cleanup.")