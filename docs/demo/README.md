# Build with AI demo video

[`build-with-ai-pd-model.mp4`](build-with-ai-pd-model.mp4) (about 1.5 minutes) shows the whole flow:

1. Add a **Demo credit data (PD/LGD/CCF)** block and run it.
2. Tag column roles: `default_flag` as target, `application_id` as ID, `reference_date` as date, and the post-default columns as excluded.
3. Select the block, click **Build with AI**, and describe a logistic-regression PD model.
4. Review the plan, which shows as ghost blocks on the canvas, then approve it.
5. The AI builds and runs each block: split, WoE binning, logistic regression, scoring, Gini/KS/PSI.
6. Read the build report.

The AI's planning and building are fast-forwarded (about 14x, marked by a purple
"Fast-forward" badge). Everything else plays in real time.

## Re-recording

```bash
modelmaker-api                                   # in an empty folder, with the claude CLI logged in
node docs/demo/record.js                         # needs playwright; writes raw.webm
python docs/demo/speedup.py raw.webm             # needs ffmpeg; writes demo.mp4
```

`speedup.py` finds the fast-forward stretches by detecting the badge in the video,
so it doesn't depend on wall-clock timestamps. The AI's plan differs from run to
run, so each recording comes out slightly different.
