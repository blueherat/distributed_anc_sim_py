#!/usr/bin/env python3
import sys
import json
import h5py
from pathlib import Path


def safe_bytes(x):
    try:
        return x.decode('utf-8')
    except Exception:
        return str(x)


def main(pth):
    p = Path(pth)
    if not p.exists():
        print(f"NOT_FOUND:{pth}")
        return 2

    with h5py.File(str(p), 'r') as f:
        # try to get n_rooms
        n_rooms = None
        if 'raw/room_params/room_size' in f:
            n_rooms = int(f['raw/room_params/room_size'].shape[0])

        datasets = {}

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = {
                    'shape': tuple(obj.shape),
                    'dtype': str(obj.dtype),
                }

        f.visititems(visitor)

        print('FILE:', str(p))
        print('NUM_SAMPLES (inferred):', n_rooms)
        # root attrs
        if len(f.attrs) > 0:
            print('\nROOT ATTRIBUTES:')
            for k, v in f.attrs.items():
                if isinstance(v, (bytes, bytearray)):
                    try:
                        s = v.decode('utf-8')
                    except Exception:
                        s = str(v)
                    print(f" - {k}: {s}")
                else:
                    print(f" - {k}: {v}")

        print('\n--- Per-sample datasets (leading dim == num_samples) ---')
        per_sample = []
        for name, info in datasets.items():
            shape = info['shape']
            if n_rooms is not None and len(shape) > 0 and shape[0] == n_rooms:
                per_sample.append((name, info))

        if not per_sample:
            print(' (none detected with leading dim == inferred sample count)')
        else:
            for name, info in sorted(per_sample):
                print(f" - {name} : per-sample shape={info['shape'][1:]}, dtype={info['dtype']}")

        print('\n--- Processed/group-level datasets ---')
        for name, info in sorted(datasets.items()):
            if name.startswith('processed/') or not name.startswith('raw/'):
                print(f" - {name} : shape={info['shape']}, dtype={info['dtype']}")

        print('\n--- All datasets (full list) ---')
        for name, info in sorted(datasets.items()):
            print(f" - {name} : shape={info['shape']}, dtype={info['dtype']}")

    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: inspect_h5_seeded.py <h5_path>')
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
