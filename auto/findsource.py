import os

# Define a list of common source code file extensions
SOURCE_CODE_EXTENSIONS = ['.py', '.java', '.js', '.cpp', '.c', '.cs', '.rb', '.go', '.php'] 
                         # '.html', '.css', '.ts']

def find_source_code_files(directory, max_depth=None):
    """
    Recursively searches for source code files in a directory and subdirectories.

    Args:
        directory (str): The root directory to search.
        max_depth (int, optional): Maximum depth of subdirectories to search. If None, no limit.

    Returns:
        List of file paths that match the source code extensions.
    """
    source_files = []
    base_depth = directory.rstrip(os.sep).count(os.sep)  # Base directory depth

    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(directory):
        current_depth = root.count(os.sep) - base_depth

        # If max_depth is specified, skip directories deeper than max_depth
        if max_depth is not None and current_depth > max_depth:
            continue

        for file in files:
            # Get the file extension
            file_extension = os.path.splitext(file)[1]

            # If the file has a source code extension, add it to the list
            if file_extension in SOURCE_CODE_EXTENSIONS:
                full_path = os.path.join(root, file)
                source_files.append(full_path)
    
    return source_files