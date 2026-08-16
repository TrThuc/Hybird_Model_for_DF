from PIL import Image, ImageDraw, ImageFont
import os
import glob


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "."

OUTPUT_FILE = "figure_BAD_GAD_average.jpg"

# Kích thước mỗi ảnh trong figure
IMG_WIDTH = 384
IMG_HEIGHT = 384

# Khoảng cách giữa các cột
COLUMN_GAP = 0

# Khoảng cách giữa các hàng
ROW_GAP = 0

# Chiều cao vùng tiêu đề
HEADER_HEIGHT = 80

# Màu nền
BACKGROUND = "white"

# Tiêu đề
TITLES = [
    "Fake",
    "BAD",
    "GAD",
    "Average"
]

# Thư mục
FOLDERS = [
    "fake_2",
    "bad_2",
    "gad_2",
    "average_2"
]


# ============================================================
# FONT
# ============================================================

def get_font(size):
    possible_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "arialbd.ttf"
    ]

    for font_path in possible_fonts:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size)

    return ImageFont.load_default()


TITLE_FONT = get_font(38)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(
        (IMG_WIDTH, IMG_HEIGHT),
        Image.Resampling.LANCZOS
    )
    return img


# ============================================================
# GET FILES
# ============================================================

def get_images(folder):
    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.webp"
    ]

    files = []

    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))

    # sort để 01, 02, 03... đúng thứ tự
    files = sorted(files)

    return files


# ============================================================
# MAIN
# ============================================================

def main():

    all_images = []

    for folder in FOLDERS:

        folder_path = os.path.join(INPUT_DIR, folder)

        if not os.path.exists(folder_path):
            raise FileNotFoundError(
                f"Không tìm thấy thư mục: {folder_path}"
            )

        images = get_images(folder_path)

        if len(images) == 0:
            raise RuntimeError(
                f"Không có ảnh trong thư mục: {folder_path}"
            )

        all_images.append(images)

    # --------------------------------------------------------
    # Số hàng
    # --------------------------------------------------------

    num_rows = max(len(x) for x in all_images)

    # --------------------------------------------------------
    # Kích thước canvas
    # --------------------------------------------------------

    total_width = (
        4 * IMG_WIDTH
        + 3 * COLUMN_GAP
    )

    total_height = (
        HEADER_HEIGHT
        + num_rows * IMG_HEIGHT
        + (num_rows - 1) * ROW_GAP
    )

    canvas = Image.new(
        "RGB",
        (total_width, total_height),
        BACKGROUND
    )

    draw = ImageDraw.Draw(canvas)

    # --------------------------------------------------------
    # Vẽ tiêu đề
    # --------------------------------------------------------

    for col, title in enumerate(TITLES):

        x0 = col * (IMG_WIDTH + COLUMN_GAP)

        bbox = draw.textbbox(
            (0, 0),
            title,
            font=TITLE_FONT
        )

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        text_x = (
            x0
            + (IMG_WIDTH - text_width) / 2
        )

        text_y = (
            HEADER_HEIGHT
            - text_height
        ) / 2 - 5

        draw.text(
            (text_x, text_y),
            title,
            fill="black",
            font=TITLE_FONT
        )

    # --------------------------------------------------------
    # Ghép ảnh
    # --------------------------------------------------------

    for row in range(num_rows):

        for col in range(4):

            if row >= len(all_images[col]):
                continue

            img_path = all_images[col][row]

            img = load_image(img_path)

            x = col * (IMG_WIDTH + COLUMN_GAP)

            y = (
                HEADER_HEIGHT
                + row * (IMG_HEIGHT + ROW_GAP)
            )

            canvas.paste(img, (x, y))

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    canvas.save(
        OUTPUT_FILE,
        quality=95
    )

    print("=" * 60)
    print("Đã tạo figure:")
    print(OUTPUT_FILE)
    print(f"Kích thước: {canvas.width} x {canvas.height}")
    print(f"Số hàng: {num_rows}")
    print("=" * 60)


if __name__ == "__main__":
    main()