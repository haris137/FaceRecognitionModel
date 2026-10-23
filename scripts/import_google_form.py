"""Import volunteer submissions from a Google Form (with file-upload questions) into raw/.

Run in Colab (needs Google auth to download the uploaded files by ID):
  python scripts/import_google_form.py --responses responses.csv --raw /content/drive/MyDrive/fr_data/raw

Expected form columns (rename with the --col_* flags if yours differ):
  "Participant code"       short code you give each volunteer, e.g. P017 (NO real names in the dataset)
  "I consent"              must contain "yes"/"agree"; rows without consent are skipped
  "Gender", "Age group", "Province / region", "Skin tone"
  upload questions          any column whose header contains --upload_keyword (default "upload");
                            its cells are comma-separated Drive links
Only participants who ticked consent are imported; a withdrawal = delete raw/<code>/ and re-run prepare_data.
"""
import argparse
import io
import os
import re

import pandas as pd


def drive_service():
    try:
        from google.colab import auth
        auth.authenticate_user()
    except ImportError:
        pass
    from googleapiclient.discovery import build
    return build("drive", "v3")


def download(service, file_id, dst_dir):
    from googleapiclient.http import MediaIoBaseDownload
    meta = service.files().get(fileId=file_id, fields="name").execute()
    name = re.sub(r"[^\w.\-]+", "_", meta["name"])
    path = os.path.join(dst_dir, name)
    if os.path.exists(path):
        return path
    req = service.files().get_media(fileId=file_id)
    with io.FileIO(path, "wb") as fh:
        dl = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, help="CSV exported from the form's Sheet")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--col_id", default="Participant code")
    ap.add_argument("--col_consent", default="I consent")
    ap.add_argument("--col_gender", default="Gender")
    ap.add_argument("--col_age", default="Age group")
    ap.add_argument("--col_region", default="Province / region")
    ap.add_argument("--col_skin", default="Skin tone")
    ap.add_argument("--upload_keyword", default="upload")
    args = ap.parse_args()

    df = pd.read_csv(args.responses, dtype=str).fillna("")
    up_cols = [c for c in df.columns if args.upload_keyword.lower() in c.lower()]
    print("upload columns:", up_cols)
    svc = drive_service()
    rows = []
    for _, r in df.iterrows():
        code = re.sub(r"[^\w\-]", "", r.get(args.col_id, "")).upper()
        if not code:
            continue
        if not re.search(r"yes|agree|consent", r.get(args.col_consent, ""), re.I):
            print(f"skip {code}: no consent")
            continue
        dst = os.path.join(args.raw, code)
        os.makedirs(dst, exist_ok=True)
        n = 0
        for c in up_cols:
            for fid in re.findall(r"id=([\w-]+)", r[c]) + re.findall(r"/d/([\w-]+)", r[c]):
                try:
                    download(svc, fid, dst)
                    n += 1
                except Exception as e:  # keep going on a single bad file
                    print(f"  {code}: failed {fid}: {e}")
        rows.append({"identity": code, "gender": r.get(args.col_gender, "unknown") or "unknown",
                     "age_group": r.get(args.col_age, "unknown") or "unknown",
                     "region": r.get(args.col_region, "unknown") or "unknown",
                     "skin_tone": r.get(args.col_skin, "unknown") or "unknown",
                     "source": "volunteer", "consent_id": f"form:{r.get('Timestamp', '')}"})
        print(f"{code}: {n} files")
    ids_path = os.path.join(args.raw, "identities.csv")
    new = pd.DataFrame(rows)
    if os.path.exists(ids_path):
        old = pd.read_csv(ids_path, dtype=str)
        new = pd.concat([old[~old["identity"].isin(new["identity"])], new], ignore_index=True)
    new.to_csv(ids_path, index=False)
    print(f"{len(rows)} consenting participants imported")


if __name__ == "__main__":
    main()
