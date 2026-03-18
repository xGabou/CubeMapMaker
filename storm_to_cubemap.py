import math
import argparse

import numpy as np
from PIL import Image, ImageFilter


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def save_image(path: str, arr: np.ndarray) -> None:
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def resize_image(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(img).astype(np.float32) / 255.0


def blur_image(arr: np.ndarray, radius: float) -> np.ndarray:
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(img).astype(np.float32) / 255.0


def bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w, _ = image.shape

    x = np.clip(x, 0.0, w - 1.0)
    y = np.clip(y, 0.0, h - 1.0)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    dx = (x - x0)[..., None]
    dy = (y - y0)[..., None]

    c00 = image[y0, x0]
    c10 = image[y0, x1]
    c01 = image[y1, x0]
    c11 = image[y1, x1]

    c0 = c00 * (1.0 - dx) + c10 * dx
    c1 = c01 * (1.0 - dx) + c11 * dx
    return c0 * (1.0 - dy) + c1 * dy


def center_crop_square(arr: np.ndarray) -> np.ndarray:
    h, w, _ = arr.shape
    size = min(h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return arr[y0:y0 + size, x0:x0 + size]


def direction_from_face_uv(face: str, u: np.ndarray, v: np.ndarray):
    if face == "FRONT":
        x = u
        y = -v
        z = np.ones_like(u)

    elif face == "BACK":
        x = -u
        y = -v
        z = -np.ones_like(u)

    elif face == "RIGHT":
        x = np.ones_like(u)
        y = -v
        z = -u

    elif face == "LEFT":
        x = -np.ones_like(u)
        y = -v
        z = u

    elif face == "TOP":
        x = u
        y = np.ones_like(u)
        z = v

    elif face == "BOTTOM":
        x = u
        y = -np.ones_like(u)
        z = -v

    else:
        raise ValueError(f"Unknown face: {face}")

    length = np.sqrt(x * x + y * y + z * z)
    return x / length, y / length, z / length


def rotate_yaw_pitch_roll(x, y, z, yaw_deg, pitch_deg, roll_deg):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    x1 = cy * x + sy * z
    y1 = y
    z1 = -sy * x + cy * z

    x2 = x1
    y2 = cp * y1 - sp * z1
    z2 = sp * y1 + cp * z1

    x3 = cr * x2 - sr * y2
    y3 = sr * x2 + cr * y2
    z3 = z2

    return x3, y3, z3


def build_basis_from_forward(forward):
    f = np.array(forward, dtype=np.float32)
    f = f / np.linalg.norm(f)

    up_guess = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(np.dot(f, up_guess)) > 0.95:
        up_guess = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    right = np.cross(up_guess, f)
    right = right / np.linalg.norm(right)

    up = np.cross(f, right)
    up = up / np.linalg.norm(up)

    return right, up, f


def make_radial_profile(source_square: np.ndarray, samples: int = 2048) -> np.ndarray:
    h, w, _ = source_square.shape
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    max_r = min(cx, cy)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max_r
    rr = np.clip(rr, 0.0, 1.0)

    profile = np.zeros((samples, 3), dtype=np.float32)
    counts = np.zeros((samples, 1), dtype=np.float32)

    idx = np.clip((rr * (samples - 1)).astype(np.int32), 0, samples - 1)

    flat_idx = idx.reshape(-1)
    flat_rgb = source_square.reshape(-1, 3)

    np.add.at(profile, flat_idx, flat_rgb)
    np.add.at(counts, flat_idx, 1.0)

    counts = np.maximum(counts, 1.0)
    profile /= counts

    kernel = np.ones(21, dtype=np.float32) / 21.0
    for c in range(3):
        profile[:, c] = np.convolve(profile[:, c], kernel, mode="same")

    return profile


def sample_radial_profile(profile: np.ndarray, r: np.ndarray) -> np.ndarray:
    n = profile.shape[0]
    t = np.clip(r, 0.0, 1.0) * (n - 1)

    i0 = np.floor(t).astype(np.int32)
    i1 = np.clip(i0 + 1, 0, n - 1)
    a = (t - i0)[..., None]

    return profile[i0] * (1.0 - a) + profile[i1] * a


def sample_omnidirectional_storm(
    source_square: np.ndarray,
    radial_profile: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    radial_power: float = 1.0,
    back_blend_start: float = 0.68,
    back_blend_end: float = 0.86
) -> np.ndarray:
    x, y, z = rotate_yaw_pitch_roll(x, y, z, yaw_deg, pitch_deg, roll_deg)

    right, up, forward = build_basis_from_forward([0.0, 0.0, 1.0])

    dot_f = np.clip(x * forward[0] + y * forward[1] + z * forward[2], -1.0, 1.0)
    theta = np.arccos(dot_f)

    proj_x = x * right[0] + y * right[1] + z * right[2]
    proj_y = x * up[0] + y * up[1] + z * up[2]
    phi = np.arctan2(proj_y, proj_x)

    r = theta / math.pi
    r = np.clip(r, 0.0, 1.0)
    r = r ** radial_power

    u = 0.5 + 0.5 * r * np.cos(phi)
    v = 0.5 - 0.5 * r * np.sin(phi)

    h, w, _ = source_square.shape
    px = u * (w - 1)
    py = v * (h - 1)

    angular_sample = bilinear_sample(source_square, px, py)
    radial_sample = sample_radial_profile(radial_profile, r)

    back_t = np.clip((r - back_blend_start) / max(back_blend_end - back_blend_start, 1e-6), 0.0, 1.0)
    back_t = back_t * back_t * (3.0 - 2.0 * back_t)
    back_t = back_t[..., None]

    out = angular_sample * (1.0 - back_t) + radial_sample * back_t
    return np.clip(out, 0.0, 1.0)


def apply_face_shading(face_name: str, image: np.ndarray) -> np.ndarray:
    if face_name == "TOP":
        factor = 1.08
    elif face_name == "BOTTOM":
        factor = 1.03
    elif face_name in ("FRONT", "BACK"):
        factor = 1.0
    else:
        factor = 0.92

    return np.clip(image * factor, 0.0, 1.0)


def render_face(
    source_square: np.ndarray,
    radial_profile: np.ndarray,
    face: str,
    size: int,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    radial_power: float,
    shade_faces: bool
) -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, size, endpoint=False) + (1.0 / size)
    uu, vv = np.meshgrid(grid, grid)

    x, y, z = direction_from_face_uv(face, uu, vv)

    out = sample_omnidirectional_storm(
        source_square=source_square,
        radial_profile=radial_profile,
        x=x,
        y=y,
        z=z,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        radial_power=radial_power
    )

    if shade_faces:
        out = apply_face_shading(face, out)

    return out


def solve_harmonic_extension(
    boundary_delta: np.ndarray,
    locked: np.ndarray,
    iterations: int
) -> np.ndarray:
    field = boundary_delta.copy()

    for _ in range(iterations):
        prev = field.copy()
        field[1:-1, 1:-1, :] = (
            prev[:-2, 1:-1, :] +
            prev[2:, 1:-1, :] +
            prev[1:-1, :-2, :] +
            prev[1:-1, 2:, :]
        ) * 0.25
        field[locked] = boundary_delta[locked]

    return field

def edge_falloff(distance: np.ndarray, fade: float) -> np.ndarray:
    if fade <= 1e-6:
        return np.zeros_like(distance, dtype=np.float32)
    t = np.clip(distance / fade, 0.0, 1.0)
    return 1.0 - smoothstep01(t)


def smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def synthesize_back_face(
    front, left, right, top, bottom,
    lock_ratio=0.008,
    source_ratio=0.35,
    feather_ratio=0.24,
    source_blur_radius=0.2,
    seam_blur_radius=0.3,
) -> np.ndarray:
    size = front.shape[0]
    lock = max(2, int(size * lock_ratio))
    source_w = max(lock + 16, int(size * source_ratio))
    feather = max(lock + 12, int(size * feather_ratio))
    corner_r = feather * 1.4

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    left_src  = blur_image(right[:, -source_w:, :].copy(), source_blur_radius)
    right_src = blur_image(left[:, :source_w, :].copy(),   source_blur_radius)
    top_src   = blur_image(top[-source_w:, :, :].copy(),   source_blur_radius)
    bot_src   = blur_image(bottom[:source_w, :, :].copy(), source_blur_radius)

    left_band   = resize_image(left_src,  feather, size)
    right_band  = resize_image(right_src, feather, size)
    top_band    = resize_image(top_src,   size, feather)
    bottom_band = resize_image(bot_src,   size, feather)

    w_left   = 1.0 - smoothstep01(xx / feather)
    w_right  = 1.0 - smoothstep01(((size - 1) - xx) / feather)
    w_top    = 1.0 - smoothstep01(yy / feather)
    w_bottom = 1.0 - smoothstep01(((size - 1) - yy) / feather)

    w_corner_tl = 1.0 - smoothstep01(np.sqrt(xx**2 + yy**2) / corner_r)
    w_corner_tr = 1.0 - smoothstep01(np.sqrt((size-1-xx)**2 + yy**2) / corner_r)
    w_corner_bl = 1.0 - smoothstep01(np.sqrt(xx**2 + (size-1-yy)**2) / corner_r)
    w_corner_br = 1.0 - smoothstep01(np.sqrt((size-1-xx)**2 + (size-1-yy)**2) / corner_r)

    left_canvas   = np.zeros((size, size, 3), dtype=np.float32)
    right_canvas  = np.zeros((size, size, 3), dtype=np.float32)
    top_canvas    = np.zeros((size, size, 3), dtype=np.float32)
    bottom_canvas = np.zeros((size, size, 3), dtype=np.float32)

    left_canvas[:, :feather, :]    = left_band
    right_canvas[:, -feather:, :]  = right_band
    top_canvas[:feather, :, :]     = top_band
    bottom_canvas[-feather:, :, :] = bottom_band

    acc  = np.zeros((size, size, 3), dtype=np.float32)
    wsum = np.zeros((size, size),    dtype=np.float32)

    for canvas, w in [
        (left_canvas,   w_left),
        (right_canvas,  w_right),
        (top_canvas,    w_top),
        (bottom_canvas, w_bottom),
    ]:
        acc  += canvas * w[..., None]
        wsum += w

    corner_tl = (left_canvas  * w_left[..., None]  + top_canvas    * w_top[..., None])    * 0.5
    corner_tr = (right_canvas * w_right[..., None] + top_canvas    * w_top[..., None])    * 0.5
    corner_bl = (left_canvas  * w_left[..., None]  + bottom_canvas * w_bottom[..., None]) * 0.5
    corner_br = (right_canvas * w_right[..., None] + bottom_canvas * w_bottom[..., None]) * 0.5

    acc  += corner_tl * w_corner_tl[..., None]
    acc  += corner_tr * w_corner_tr[..., None]
    acc  += corner_bl * w_corner_bl[..., None]
    acc  += corner_br * w_corner_br[..., None]
    wsum += w_corner_tl + w_corner_tr + w_corner_bl + w_corner_br

    # Edge blend strength (0 = pure interior, 1 = pure edge band)
    edge_strength = np.clip(wsum, 0.0, 1.0)[..., None]
    edge_color = acc / np.maximum(wsum[..., None], 1e-6)

    # Interior = rotated front (real texture, no grey)
    rotated_front = np.rot90(front, 2).copy()

    # Blend: edge bands win at borders, rotated front fills the center
    back = rotated_front * (1.0 - edge_strength) + edge_color * edge_strength

    if seam_blur_radius > 0.0:
        seam_mask = np.maximum.reduce([w_left, w_right, w_top, w_bottom])[..., None]
        seam_mask = smoothstep01(seam_mask)
        blurred = blur_image(back, seam_blur_radius)
        back = back * (1.0 - seam_mask) + blurred * seam_mask

    # Pin exact border pixels
    back[:, :lock, :]  = right[:, -lock:, :]
    back[:, -lock:, :] = left[:, :lock, :]
    back[:lock, :, :]  = top[-lock:, :, :]
    back[-lock:, :, :] = bottom[:lock, :, :]

        # Final post-process: blend real neighbor pixels into the back face borders
    back = blend_back_borders(back, left, right, top, bottom, blend_width=80)

    return np.clip(back, 0.0, 1.0)


def save_debug_back_preview(
    path: str,
    back: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray
) -> None:
    size = back.shape[0]
    canvas = np.zeros((size * 2, size * 3, 3), dtype=np.float32)

    canvas[0:size, size:size * 2] = top
    canvas[size:size * 2, 0:size] = left
    canvas[size:size * 2, size:size * 2] = back
    canvas[size:size * 2, size * 2:size * 3] = right

    save_image(path, canvas)


def assemble_layout(left, front, right, back, top, bottom) -> np.ndarray:
    size = left.shape[0]
    canvas = np.zeros((size * 2, size * 4, 3), dtype=np.float32)

    canvas[0:size, size:size * 2] = top
    canvas[0:size, size * 2:size * 3] = bottom

    canvas[size:size * 2, 0:size] = left
    canvas[size:size * 2, size:size * 2] = front
    canvas[size:size * 2, size * 2:size * 3] = right
    canvas[size:size * 2, size * 3:size * 4] = back

    return canvas

def blend_back_borders(
    back: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray,
    blend_width: int = 80        # how many pixels deep the blend goes
) -> np.ndarray:
    out = back.copy()
    size = back.shape[0]

    # Build a 1D gradient: 1.0 at the very edge, 0.0 at blend_width depth
    # Using smoothstep so the falloff is perceptually smooth
    t = np.linspace(1.0, 0.0, blend_width, dtype=np.float32)
    t = t * t * (3.0 - 2.0 * t)  # smoothstep

    # LEFT border of back face → sample from right face's right edge column strip
    for i in range(blend_width):
        alpha = t[i]
        # grab column i of the neighbor (their rightmost strip maps to back's left)
        neighbor_col = right[:, -(blend_width - i), :]
        out[:, i, :] = out[:, i, :] * (1.0 - alpha) + neighbor_col * alpha

    # RIGHT border of back face → sample from left face's left edge column strip
    for i in range(blend_width):
        alpha = t[i]
        neighbor_col = left[:, blend_width - i - 1, :]
        out[:, size - 1 - i, :] = out[:, size - 1 - i, :] * (1.0 - alpha) + neighbor_col * alpha

    # TOP border of back face → sample from top face's bottom row strip
    for i in range(blend_width):
        alpha = t[i]
        neighbor_row = top[-(blend_width - i), :, :]
        out[i, :, :] = out[i, :, :] * (1.0 - alpha) + neighbor_row * alpha

    # BOTTOM border of back face → sample from bottom face's top row strip
    for i in range(blend_width):
        alpha = t[i]
        neighbor_row = bottom[blend_width - i - 1, :, :]
        out[size - 1 - i, :, :] = out[size - 1 - i, :, :] * (1.0 - alpha) + neighbor_row * alpha

    return np.clip(out, 0.0, 1.0)



def main() -> None:
    parser = argparse.ArgumentParser(description="Render a continuous storm cubemap on all faces.")
    parser.add_argument("input", help="Path to source image")
    parser.add_argument("output", help="Path to output PNG")
    parser.add_argument("--face-size", type=int, default=1024, help="Size of each cube face")
    parser.add_argument("--work-size", type=int, default=2048, help="Internal square source size")
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw rotation in degrees")
    parser.add_argument("--pitch", type=float, default=0.0, help="Pitch rotation in degrees")
    parser.add_argument("--roll", type=float, default=0.0, help="Roll rotation in degrees")
    parser.add_argument("--radial-power", type=float, default=0.82, help="Lower values spread the storm more across all faces")
    parser.add_argument("--no-face-shading", action="store_true", help="Disable per face brightness variation")
    parser.add_argument("--back-strip-ratio", type=float, default=0.14, help="Thickness of seam matching border on the back face")
    parser.add_argument("--back-iterations", type=int, default=500, help="Number of inward correction diffusion iterations for the back face")
    parser.add_argument("--back-interior-blur", type=float, default=0.6, help="Tiny blur only on the interior of the corrected back face")
    parser.add_argument("--debug-back-preview", default=None, help="Optional path to save a preview of the synthesized BACK face with neighbors")

    args = parser.parse_args()

    source = load_image(args.input)
    source_square = center_crop_square(source)
    source_square = resize_image(source_square, args.work_size, args.work_size)
    radial_profile = make_radial_profile(source_square)

    left = render_face(source_square, radial_profile, "LEFT", args.face_size, args.yaw, args.pitch, args.roll, args.radial_power, not args.no_face_shading)
    front = render_face(source_square, radial_profile, "FRONT", args.face_size, args.yaw, args.pitch, args.roll, args.radial_power, not args.no_face_shading)
    right = render_face(source_square, radial_profile, "RIGHT", args.face_size, args.yaw, args.pitch, args.roll, args.radial_power, not args.no_face_shading)
    top = render_face(source_square, radial_profile, "TOP", args.face_size, args.yaw, args.pitch, args.roll, args.radial_power, not args.no_face_shading)
    bottom = render_face(source_square, radial_profile, "BOTTOM", args.face_size, args.yaw, args.pitch, args.roll, args.radial_power, not args.no_face_shading)
    
    back = synthesize_back_face(
        front=front, left=left, right=right, top=top, bottom=bottom
    )







    if args.debug_back_preview:
        save_debug_back_preview(args.debug_back_preview, back, left, right, top, bottom)

    final = assemble_layout(left, front, right, back, top, bottom)
    save_image(args.output, final)

    print(f"Saved cubemap layout to: {args.output}")


if __name__ == "__main__":
    main()