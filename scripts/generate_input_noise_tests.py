"""Generate LISO input noise validation scripts from existing scripts that specify output noise.

Note that some scripts should not be input-referred, such as those with floating inputs
(unsupported by LISO). These should be identified and removed manually after this script creates
them.

Generated files can be removed with e.g. `find . -name \*-input-referred.fil -type f -delete`
(replacing `-input-referred` with whatever the specified suffix is).
"""

import os
from pathlib import Path
import glob
import re

PARENT_DIR = Path(__file__).resolve().parent.parent
LISO_SCRIPT_DIR = PARENT_DIR / "tests" / "scripts" / "liso"

def generate_input_noise_script(script, overwrite=False, suffix="-input-referred"):
    """Generate an input noise version of the specified LISO script."""
    re_flags = re.IGNORECASE | re.MULTILINE
    noise_regex = r'^noise\s(.+)\s(.+).*$'
    # Create target file path.
    pieces = os.path.splitext(script)
    target_file = pieces[0] + suffix + pieces[1]
    # Check for existing file.
    if not overwrite and os.path.exists(target_file):
        print(f"Skipping {script} (target file already exists)")
        return False
    with open(script, "r") as fobj:
        text = fobj.read()
    # Find the noise command.
    match = re.search(noise_regex, text, flags=re_flags)
    if match is None:
        print(f"Skipping {script} (no noise command)")
        return False
    # Replace noise with input noise
    text = text[:match.start()] + "inputnoise" + text[match.start()+5:]
    # Write new script to file.
    with open(target_file, "w") as fobj:
        fobj.write(text)
    print(f"Wrote {target_file}")
    return True


if __name__ == "__main__":
    count = 0
    for script in glob.glob(os.path.join(LISO_SCRIPT_DIR, "**/*.fil"), recursive=True):
        if generate_input_noise_script(script):
            count += 1
    print(f"Wrote {count} files.")
