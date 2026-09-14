import wfdb

print("Downloading MIT-BIH Arrhythmia Database...")

wfdb.dl_database(
    "mitdb",
    dl_dir="data/mitdb"
)

print("Download complete!")