from unidiff import PatchSet


def analyze_diff_path_set(patch_set):
    print_statements = []

    # Parse the hunk header to extract line numbers and changes
    for patched_file in patch_set:
        for hunk in patched_file:
            # Extract the line information from the hunk header
            old_start, old_lines = hunk.source_start, hunk.source_length
            new_start, new_lines = hunk.target_start, hunk.target_length
            print_statements.append(
                f"Changes start at line {old_start} in the original file and line {new_start} in the new file."
            )
            print_statements.append(
                f"{old_lines} lines were modified in the original file, and {new_lines} lines were modified in the new file."
            )

    # Count the number of added and removed lines
    added_lines = patch_set.added
    removed_lines = patch_set.removed

    print_statements.append(f"Total added lines: {added_lines}")
    print_statements.append(f"Total removed lines: {removed_lines}")

    return print_statements


def create_patch_set(patch):
    if not patch.startswith("---") or not patch.startswith("+++"):
        patch = f"--- a/file\n+++ b/file\n{patch}"

    # Create the PatchSet object
    patch_set = PatchSet.from_string(patch)
    return patch_set
