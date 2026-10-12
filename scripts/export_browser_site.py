"""Copy the standalone static site into a GitHub Pages checkout.

This copies only public frontend assets and licenses. It does not commit,
push, remove unrelated files, or include local checkpoints and training data.
"""
import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    public = ['index.html', 'style.css', 'app.js', 'viewer.js', 'motion.js',
              'worker.js', 'tokenizer.js', 'settings.json', 'assets/examples.json']
    args.output.mkdir(parents=True, exist_ok=True)
    for name in public:
        source, target = root / 'web' / name, args.output / name
        if source.is_symlink():
            raise ValueError('Public site asset is a symlink: ' + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name in ['LICENSE', 'NOTICE', 'licenses/mjswan-Apache-2.0.txt']:
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    (args.output / '.nojekyll').touch()
    print('Exported browser demo to ' + str(args.output.resolve()))


if __name__ == '__main__':
    main()
