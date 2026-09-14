"""
Generates a tiny synthetic dataset laid out exactly like Something-Something-v2
(same directory structure and labels/*.json schema that SomethingDataset in
something_dataloader.py expects), so the real SSv2 dataloader code path can be
exercised without downloading the ~20GB+ real corpus.

Unlike generate_synthetic_ssv2.py (ffmpeg `testsrc2`, whose *whole frame*
animates), this generator produces FAITHFUL low-rank motion: a single bright,
internally-textured object moves with a constant per-clip velocity over a
*static* textured background, so only the moving region changes between frames.
That matches TDV's stated assumption that the RGB difference Δx = x_{t+1} - x_t
is "intrinsically lower rank than the frames themselves, as the background scene
pixels remain largely unchanged ... and only moving regions contribute a
non-zero signal" (paper §3.2). With real, learnable motion the next-frame
prediction task is non-trivial, so the motion encoder + MSE loss have something
to do — which is exactly what the Table-4 collapse ablation probes.

This is a pipeline/reproduction stand-in for real video content, not a claim of
semantic realism. Two nominal "classes" (horizontal- vs vertical-dominant
motion) exist only to satisfy the labels/*.json schema; the reproduction signal
does not use the labels.
"""
import argparse
import json
import os
import subprocess

import numpy as np

# NOTE: SomethingDataset maps a clip to its label by stripping "[" and "]" from
# each record's `template` and looking the result up in labels.json, so the
# bracket-stripped template must equal the class key — hence classes start with
# "something " and template is "[something] <rest>" (mirrors generate_synthetic_ssv2.py).
CLASSES = [
	"something moving mostly horizontally",
	"something moving mostly vertically",
]


def _smooth_background(rng, size):
	"""Static low-frequency RGB texture: upsample small random noise."""
	small = rng.random((8, 8, 3))
	# nearest-neighbour upsample to size x size, then a cheap box blur
	reps = size // 8 + 1
	big = np.kron(small, np.ones((reps, reps, 1)))[:size, :size, :]
	# 3x3 box blur (a couple passes) to remove hard block edges
	for _ in range(2):
		big = (
			big
			+ np.roll(big, 1, 0) + np.roll(big, -1, 0)
			+ np.roll(big, 1, 1) + np.roll(big, -1, 1)
		) / 5.0
	return (0.25 + 0.4 * big)  # keep background dim so the object stands out


def _object_sprite(rng, radius):
	"""A bright, internally-textured filled disc (RGBA-ish: rgb + alpha mask)."""
	d = 2 * radius + 1
	yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
	mask = (xx * xx + yy * yy) <= radius * radius
	base = np.array([rng.uniform(0.6, 1.0), rng.uniform(0.6, 1.0), rng.uniform(0.6, 1.0)])
	# radial texture so the encoder has real content to represent, not a flat blob
	tex = 0.6 + 0.4 * np.cos(np.sqrt(xx * xx + yy * yy) / max(radius, 1) * 3.0)
	rgb = np.clip(base[None, None, :] * tex[:, :, None], 0, 1)
	return rgb, mask


def make_clip_frames(rng, size, n_frames, horizontal):
	bg = _smooth_background(rng, size)
	radius = rng.integers(size // 10, size // 6)
	rgb, mask = _object_sprite(rng, radius)

	# constant velocity; class controls the dominant axis, sign/rate randomised
	speed = rng.uniform(6.0, 10.0)  # px per frame -> clear displacement over a 0.25s (=3 frame) gap
	if horizontal:
		vx, vy = speed * rng.choice([-1, 1]), rng.uniform(-2, 2)
	else:
		vx, vy = rng.uniform(-2, 2), speed * rng.choice([-1, 1])

	x = rng.uniform(radius, size - radius)
	y = rng.uniform(radius, size - radius)

	frames = []
	for _ in range(n_frames):
		frame = bg.copy()
		cx, cy = int(round(x)), int(round(y))
		x0, x1 = cx - radius, cx + radius + 1
		y0, y1 = cy - radius, cy + radius + 1
		# clip sprite to frame bounds
		sx0, sy0 = max(0, -x0), max(0, -y0)
		x0c, y0c = max(0, x0), max(0, y0)
		x1c, y1c = min(size, x1), min(size, y1)
		mh, mw = y1c - y0c, x1c - x0c
		if mh > 0 and mw > 0:
			m = mask[sy0:sy0 + mh, sx0:sx0 + mw]
			patch = rgb[sy0:sy0 + mh, sx0:sx0 + mw]
			region = frame[y0c:y1c, x0c:x1c]
			region[m] = patch[m]
			frame[y0c:y1c, x0c:x1c] = region
		frames.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))

		# advance + bounce off walls (keeps the object on screen, motion stays low-rank)
		x += vx; y += vy
		if x < radius or x > size - radius: vx = -vx; x = np.clip(x, radius, size - radius)
		if y < radius or y > size - radius: vy = -vy; y = np.clip(y, radius, size - radius)

	return np.stack(frames)  # (T, H, W, 3) uint8


def encode_webm(frames, path, fps):
	size = frames.shape[1]
	cmd = [
		"ffmpeg", "-y", "-v", "error",
		"-f", "rawvideo", "-pix_fmt", "rgb24",
		"-s", f"{size}x{size}", "-r", str(fps),
		"-i", "-",
		"-c:v", "libvpx", "-b:v", "1M", "-pix_fmt", "yuv420p",
		path,
	]
	proc = subprocess.run(cmd, input=frames.tobytes(), stderr=subprocess.PIPE)
	if proc.returncode != 0:
		raise RuntimeError(f"ffmpeg failed for {path}: {proc.stderr.decode()[:500]}")


def generate_split(out_dir, n_per_class, size, fps, n_frames, start_id, seed0):
	video_dir = os.path.join(out_dir, "20bn-something-something-v2")
	os.makedirs(video_dir, exist_ok=True)
	records = []
	next_id = start_id
	for class_idx, template_clean in enumerate(CLASSES):
		for k in range(n_per_class):
			rng = np.random.default_rng(seed0 + next_id)
			frames = make_clip_frames(rng, size, n_frames, horizontal=(class_idx == 0))
			video_id = str(next_id); next_id += 1
			encode_webm(frames, os.path.join(video_dir, f"{video_id}.webm"), fps)
			records.append({
				"id": video_id,
				"label": template_clean,
				"template": f"[something] {template_clean.replace('something ', '')}",
				"placeholders": ["something"],
			})
	return records, next_id


def main():
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--out_dir", required=True)
	p.add_argument("--n_train_per_class", type=int, default=16)
	p.add_argument("--n_val_per_class", type=int, default=4)
	p.add_argument("--size", type=int, default=224)
	p.add_argument("--fps", type=int, default=12)
	p.add_argument("--duration", type=float, default=2.5)
	p.add_argument("--skip_if_exists", action="store_true", default=False)
	args = p.parse_args()

	labels_dir = os.path.join(args.out_dir, "labels")
	os.makedirs(labels_dir, exist_ok=True)
	if args.skip_if_exists and os.path.exists(os.path.join(labels_dir, "train.json")):
		print(f"synthetic motion dataset already exists at {args.out_dir}, skipping generation")
		return

	n_frames = int(args.fps * args.duration)
	with open(os.path.join(labels_dir, "labels.json"), "w") as f:
		json.dump({c: i for i, c in enumerate(CLASSES)}, f, indent=2)

	train, next_id = generate_split(args.out_dir, args.n_train_per_class, args.size, args.fps, n_frames, 0, seed0=1000)
	with open(os.path.join(labels_dir, "train.json"), "w") as f:
		json.dump(train, f, indent=2)
	val, _ = generate_split(args.out_dir, args.n_val_per_class, args.size, args.fps, n_frames, next_id, seed0=9000)
	with open(os.path.join(labels_dir, "validation.json"), "w") as f:
		json.dump(val, f, indent=2)

	print(f"generated synthetic LOW-RANK-MOTION dataset at {args.out_dir}: "
		  f"{len(train)} train / {len(val)} val clips ({len(CLASSES)} classes, "
		  f"{args.size}x{args.size}@{args.fps}fps, {args.duration}s, moving disc on static bg)")


if __name__ == "__main__":
	main()

