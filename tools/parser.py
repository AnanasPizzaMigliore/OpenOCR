import re
from datetime import datetime
from dateutil import parser as date_parser

class DateParser:
    def __init__(self):
        self.noise_patterns = [
            (r'[|\[\]\{\}]', ''),
            (r'[\u4e00-\u9fa5]', ' '),
            (r'[^a-zA-Z0-9\s\.\-\/:]', ' '),
            (r'(\d{4})\s*[\./-]\s*(\d{1,2})\s*[\./-]\s*(\d{1,2})', r'\1-\2-\3'),
            (r'(?<!\d)(\d{1,2})\s*[\./-]\s*(\d{1,2})\s*[\./-]\s*(\d{4})', r'\1-\2-\3'),
        ]

        self.month_map = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
            'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
            'SEP': 9, 'SEPT': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }

    # ----------------------------
    # Cleaning
    # ----------------------------
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        cleaned = text.strip()
        for pattern, replacement in self.noise_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        return cleaned.strip()

    # ----------------------------
    # Main parser
    # ----------------------------
    def parse(self, text):
        if not text:
            return None

        clean_text = self.clean_text(text)
        if len(clean_text) < 4 or not any(c.isdigit() for c in clean_text):
            return None

        now = datetime.now()

        # ==================================================
        # 1. MM.DD / DD.MM (STRICT, RETURN EARLY)
        # ==================================================
        md_match = re.match(r'^\s*(\d{1,2})[.\-/](\d{1,2})\s*$', clean_text)
        if md_match:
            a, b = int(md_match.group(1)), int(md_match.group(2))
            year = now.year

            valid_dates = []
            try:
                valid_dates.append(datetime(year, a, b))  # MM.DD
            except ValueError:
                pass

            try:
                valid_dates.append(datetime(year, b, a))  # DD.MM
            except ValueError:
                pass

            if valid_dates:
                return valid_dates[0]

        # ==================================================
        # 1.5. MM.YYYY / YYYY.MM (PARTIAL DATES -> FIRST DAY)
        # ==================================================
        # Catch MM/YYYY (e.g., 05.2024) -> Output: 2024-05-01
        my_match = re.match(r'^\s*(\d{1,2})[.\-/](\d{4})\s*$', clean_text)
        if my_match:
            m, y = int(my_match.group(1)), int(my_match.group(2))
            if 1 <= m <= 12 and 2000 <= y <= 2035:
                return datetime(y, m, 1)

        # Catch YYYY/MM (e.g., 2024.05) -> Output: 2024-05-01
        ym_match = re.match(r'^\s*(\d{4})[.\-/](\d{1,2})\s*$', clean_text)
        if ym_match:
            y, m = int(ym_match.group(1)), int(ym_match.group(2))
            if 1 <= m <= 12 and 2000 <= y <= 2035:
                return datetime(y, m, 1)

        # ==================================================
        # 2. TEXT MONTH FORMATS (STRICT, RETURN EARLY)
        # ==================================================
        upper_text = clean_text.upper()

        # DD MON YY
        m1 = re.match(r'^\s*(\d{1,2})\s+([A-Z]{3,4})\s+(\d{2})\s*$', upper_text)
        if m1:
            d, mon, y = m1.groups()
            if mon in self.month_map:
                return datetime(2000 + int(y), self.month_map[mon], int(d))

        # MON DD YY
        m2 = re.match(r'^\s*([A-Z]{3,4})\s+(\d{1,2})\s+(\d{2})\s*$', upper_text)
        if m2:
            mon, d, y = m2.groups()
            if mon in self.month_map:
                return datetime(2000 + int(y), self.month_map[mon], int(d))

        # MON/DD/YY
        m3 = re.match(r'^\s*([A-Z]{3,4})[\/\-](\d{1,2})[\/\-](\d{2})\s*$', upper_text)
        if m3:
            mon, d, y = m3.groups()
            if mon in self.month_map:
                return datetime(2000 + int(y), self.month_map[mon], int(d))

        # ==================================================
        # 2.5. YEAR-FIRST FORMATS (Strict ONLY for 4-digit years)
        # ==================================================
        # Catch YYYY.MM.DD (e.g., 2021.05.23)
        ymd_match = re.match(r'^\s*(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})\s*$', clean_text)
        if ymd_match:
            y, m, d = int(ymd_match.group(1)), int(ymd_match.group(2)), int(ymd_match.group(3))
            try:
                return datetime(y, m, d)
            except ValueError:
                pass

        # ==================================================
        # 3. DENSE NUMERIC
        # ==================================================
        candidates = []
        digits_only = re.sub(r'\D', '', clean_text)

        dense_formats = []
        if len(digits_only) == 4:
            dense_formats = ['%m%d', '%d%m']
        elif len(digits_only) == 6:
            dense_formats = ['%y%m%d', '%d%m%y', '%m%d%y']
        elif len(digits_only) == 8:
            dense_formats = ['%Y%m%d', '%d%m%Y', '%m%d%Y']

        for fmt in dense_formats:
            try:
                dt = datetime.strptime(digits_only, fmt)
                if dt.year == 1900:
                    dt = dt.replace(year=now.year)
                candidates.append(dt)
            except ValueError:
                continue

        # ==================================================
        # 4. DATEUTIL (LAST RESORT ONLY)
        # ==================================================
        for df_setting in [True, False]:
            try:
                dt = date_parser.parse(clean_text, fuzzy=True, dayfirst=df_setting)
                if dt.year < 1901:
                    dt = dt.replace(year=now.year)
                candidates.append(dt)
            except Exception:
                continue

        if not candidates:
            return None

        # ==================================================
        # 5. SELECT BEST (closest to today)
        # ==================================================
        unique = {}
        for dt in candidates:
            unique[dt.date()] = dt

        best = None
        min_diff = float('inf')

        for dt in unique.values():
            if not (2000 <= dt.year <= 2035):
                continue
            diff = abs((dt.date() - now.date()).days)
            if diff < min_diff:
                min_diff = diff
                best = dt

        return best


# ==========================================
# FILE PROCESSING
# ==========================================
def parse_dates_from_file(input_filepath, output_filepath):
    parser = DateParser()
    results = []

    with open(input_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        if not line.strip():
            continue

        parts = line.strip().split('\t', 1)
        if len(parts) != 2:
            continue

        filename, raw = parts
        
        # Get the datetime object from the parser
        parsed_dt = parser.parse(raw)

        # Break it down into discrete strings if it successfully parsed
        if parsed_dt:
            year = parsed_dt.strftime('%Y')
            month = parsed_dt.strftime('%m')
            day = parsed_dt.strftime('%d')
        else:
            year, month, day = "None", "None", "None"

        results.append(f"{filename},{raw},{year},{month},{day}")

    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write("Filename,Raw_String,Year,Month,Day\n")
        f.write("\n".join(results))

    print(f"Done! Parsed {len(results)} entries.")
    print(f"Saved to: {output_filepath}")


# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    parse_dates_from_file(
        '/scratch/penghao/datasets/Date-Real/eval/labels_fixed.txt',
        'baseline_predictions.csv'
    )