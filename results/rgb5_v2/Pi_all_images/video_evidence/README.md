# RGB5 Raspberry Pi video operation evidence

Two unique RGB source clips were processed on both exact RGB5 Hailo-8 models: four actual runs, 200 frames each (800 model-frame evaluations total). Results were returned from drone-pi with device identity, HEF hashes, source-video hashes, output hashes, counts and per-frame logs.

The website Camera and Tracking page retains its earlier computer/Pi video and shows two new HusAero-8n versus HusAero-11n comparisons beneath it. Each panel was recorded sequentially on the same Pi; comparison playback aligns source frame numbers and is not a claim of simultaneous inference or processing speed. The clips are also attached to the full Pi experiment record. H.264 encoding and side-by-side layout do not alter predicted coordinates or confidence values.

Inputs: DUT Anti-UAV visible/RGB recordings 20190926_111509_1_1 and 20190926_102042_1_5, obtained from the project's existing local test footage. Credit: original DUT Anti-UAV creators, https://github.com/wangdongdut/DUT-Anti-UAV . Source markings are retained. Full original source recordings and annotated derivatives are not redistributed in this public Git repository; the website keeps its existing access settings.

Counts are operation checks on unannotated video, not tracking accuracy or recall. Tracker: basic centre association, not ByteTrack or Lucas-Kanade. Class mistakes remain visible; neither image-test scores nor the demonstration establish reliable subtype recognition in arbitrary scenes.
