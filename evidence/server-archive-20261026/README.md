# Server retirement archive, 26 October 2026

Project: wave-cw-26. Source snapshot commit: `cb68c0518eb46e3c615529716b71a13c79f5468d`.

Release: https://github.com/handsomehs/WAVE-CW-26/releases/tag/server-archive-20261026

Current text experiment records, metadata, summaries and profiling CSV exports are preserved directly in Git. MLaS original submission ZIPs are also in Git. Large diagnostics are separate Release assets and are not included in a normal Git clone or GitHub-generated source ZIP.

`wave-cw-26-experiment-records-20261026.tar.zst` contains 24 files totalling 356678265 uncompressed bytes. The archive retains original project-relative paths under `wave-cw-26/`. archive-manifest.json records the original snapshot, file sizes, per-file SHA256 and paths outside the selected archive. FILES.sha256 verifies the extracted files. SHA256SUMS.assets verifies each other Release asset.

Download and restore from an empty working directory:

```sh
gh release download server-archive-20261026 --repo handsomehs/WAVE-CW-26
sha256sum -c SHA256SUMS.assets
tar --zstd -xf wave-cw-26-experiment-records-20261026.tar.zst
sha256sum -c FILES.sha256
```

MLaS requires GitHub authentication because its repository is private. Extract into an empty directory first, then merge with a Git clone if desired. Original file contents, filenames and basic tar permissions are preserved; no byte-for-byte image of the server filesystem is implied.

The final report ASPP_CW1.pdf, its TeX source, implementations, job manifests and recorded results are preserved in Git. The original build-dev/gpu_alloc_check.cu auxiliary test source is now tracked so the existing TESTING.md commands remain usable. Build products and installed Nsight rule templates are excluded; the empty local run log is included without implying that it contains original console results.

Preservation checks and independent download results are recorded in verification.json in this repository directory after upload. No GPU benchmarks were rerun: this change preserves existing evidence and does not change experimental claims.
