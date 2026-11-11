#!/usr/bin/env python3
"""Local RGB-video preview with Hailo inference and a basic single-target tracker.

Run on the Raspberry Pi: python3 drone_live.py
Open http://PI_ADDRESS:8080 on the same network. Ctrl+C stops the server.
Uses a prerecorded visible video, not a camera. Tracking uses detector association
and short-lived Lucas-Kanade optical flow, with flow-only boxes marked separately.
Drone subtype remains experimental; display uses drone and logs retain class IDs.
Record the annotated output with Record, then use Stop and download.
HailoRT API reference: https://github.com/hailo-ai/hailo-rpi5-examples/blob/main/community_projects/Navigator/modules/hailo.py
"""
import argparse
import hashlib
from contextlib import contextmanager
from functools import partial
from uuid import uuid4
import json
import math
import threading
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

NAMES = ['fixed_wing_drone', 'multirotor_drone', 'bird', 'airplane', 'helicopter']
ROOT = Path('/home/kwts/Drone-AI-Project')
VIDEO_ROOT = ROOT / 'Drone-AI-Project/test_videos/test'
VIDEO_NAMES = ['20190926_102042_1_5', '20190925_200805_1_6',
               '20190926_102042_1_9', '20190926_111509_1_3', '20190925_134301_1_6']


def letterbox(frame, size=640):
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh))
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb[None]), (nw / w, nh / h, left, top)


def suppress_duplicates(detections, iou_threshold=.45, containment=.90):
    """Merge spatial duplicates across the two drone subtypes, keeping actual top confidence.

    Containment handles oversized nested predictions which ordinary IoU NMS misses.
    Separated targets are retained. This does not correct a predicted subtype.
    """
    kept=[]
    for candidate in sorted(detections,key=lambda d:d[1],reverse=True):
        box=np.asarray(candidate[0],dtype=float);area=np.prod(np.maximum(0,box[2:]-box[:2]));duplicate=False
        if area<=0 or not np.isfinite([*box,candidate[1]]).all():continue
        for previous,_,_ in kept:
            other=np.asarray(previous,dtype=float);other_area=np.prod(np.maximum(0,other[2:]-other[:2]))
            inter=np.prod(np.maximum(0,np.minimum(box[2:],other[2:])-np.maximum(box[:2],other[:2])))
            overlap=inter/max(area+other_area-inter,1e-9)
            inside=inter/max(min(area,other_area),1e-9)
            distance=np.linalg.norm((box[:2]+box[2:]-other[:2]-other[2:])/2)
            close=distance<=.5*min(np.linalg.norm(box[2:]-box[:2]),np.linalg.norm(other[2:]-other[:2]))
            if overlap>=iou_threshold or (inside>=containment and close):duplicate=True;break
        if not duplicate:kept.append(candidate)
    return kept


def tile_fallback(predict, frame, detections):
    """Four overlapping views only after a full-frame miss; no target coordinates supplied."""
    if detections:return detections
    h,w=frame.shape[:2]
    if max(h,w)<=640:return detections
    tw,th=round(w*.6),round(h*.6);combined=[]
    for y in [0,h-th]:
        for x in [0,w-tw]:
            for box,confidence,cls in predict(frame[y:y+th,x:x+tw]):
                combined.append(([box[0]+x,box[1]+y,box[2]+x,box[3]+y],confidence,cls))
    return suppress_duplicates(combined)


def decode(raw, shape, transform):
    """Hailo NMS output: batch -> class -> [ymin,xmin,ymax,xmax,score]."""
    h, w = shape[:2]
    sx, sy, left, top = transform
    detections = []
    per_class = raw[0]
    if len(per_class) != 5:
        raise ValueError('Expected the five-class HusAero model; check the HEF file.')
    for class_id, rows in enumerate(per_class[:2]):  # Both drone subtypes; ignore the other three classes.
        arr = np.asarray(rows, dtype=np.float32)
        if arr.size == 0:
            continue
        if arr.ndim != 2 or arr.shape[1] != 5:
            raise ValueError('Unexpected Hailo NMS output shape: ' + str(arr.shape))
        for y1, x1, y2, x2, score in arr:
            if not np.isfinite([x1, y1, x2, y2, score]).all() or score < 0.25:
                continue
            box = [(float(x1)*640-left)/sx, (float(y1)*640-top)/sy,
                   (float(x2)*640-left)/sx, (float(y2)*640-top)/sy]
            box = [max(0, min(w-1, box[0])), max(0, min(h-1, box[1])),
                   max(0, min(w-1, box[2])), max(0, min(h-1, box[3]))]
            if box[2] > box[0] and box[3] > box[1]:
                detections.append((box, float(score), class_id))
    return suppress_duplicates(detections)



@contextmanager
def detector(backend, model_path, small_target=False):
    if backend == 'pc':
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(4)
        model = YOLO(str(model_path))
        if list(model.names.values()) != NAMES:
            raise ValueError('Expected the five-class HusAero checkpoint.')

        def predict_once(frame):
            boxes = model.predict(frame, device='cpu', imgsz=640, conf=.25,
                                  rect=False, iou=.45, agnostic_nms=True, classes=[0, 1], verbose=False)[0].boxes
            return suppress_duplicates([(box, float(score), int(cls)) for box, score, cls in
                    zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist())])

        def predict(frame):
            detections=predict_once(frame)
            return tile_fallback(predict_once,frame,detections) if small_target else detections

        yield predict
        return

    from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams,
                                InputVStreamParams, OutputVStreamParams, InferVStreams, FormatType)
    hef = HEF(str(model_path))
    input_info = hef.get_input_vstream_infos()[0]
    output_info = hef.get_output_vstream_infos()[0]
    if tuple(input_info.shape) != (640, 640, 3):
        raise ValueError('Expected 640x640x3 input, got ' + str(input_info.shape))
    with VDevice() as device:
        config = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
        group = device.configure(hef, config)[0]
        inp = InputVStreamParams.make(group, quantized=True, format_type=FormatType.UINT8)
        out = OutputVStreamParams.make(group, quantized=False, format_type=FormatType.FLOAT32)
        with InferVStreams(group, inp, out) as pipeline:
            with group.activate(group.create_params()):
                def predict_once(frame):
                    data, transform = letterbox(frame)
                    result = pipeline.infer({input_info.name: data})
                    return decode(result[output_info.name], frame.shape, transform)

                def predict(frame):
                    detections=predict_once(frame)
                    return tile_fallback(predict_once,frame,detections) if small_target else detections

                yield predict


class Tracker:
    def __init__(self, max_gap=15):
        self.max_gap = max_gap
        self.next_id = 1
        self.flow=None
        self.reset()

    def reset(self):
        self.box = None
        self.track_id = None
        self.gap = 0
        self.trail = deque(maxlen=12)
        self.flow_box=None

    def update(self, detections, shape, frame=None):
        if frame is not None:
            from flow_assisted_tracker import FlowAssistedTracker
            if self.flow is None:
                self.flow=FlowAssistedTracker(20,activation=.25,keep_threshold=.25,max_gap_seconds=.5)
            # Both subtypes belong to one tracking category; original predicted classes stay intact.
            rows=np.asarray([[*box,score,0] for box,score,_ in detections],np.float32).reshape(-1,6)
            tracks=self.flow.update(frame,rows)
            self.flow_box=None
            if not len(tracks):self.box=None;self.trail.clear();return None,self.flow.state
            self.box=tracks[0,:4].tolist();self.track_id=int(tracks[0,4])
            state=self.flow.state
            selected=None
            if state=='DETECTION_CONFIRMED' and detections:
                selected=max(range(len(detections)),key=lambda i:float(self.flow._iou(np.asarray(self.box),np.asarray([detections[i][0]]))[0]))
            elif state=='FLOW_ASSISTED':self.flow_box=self.box
            # Do not render velocity-only propagation as a measured target position.
            if selected is not None or self.flow_box is not None:
                self.trail.append((round((self.box[0]+self.box[2])/2),round((self.box[1]+self.box[3])/2)))
            return selected,state+' ID '+str(self.track_id)
        chosen = None
        if self.box is not None:
            cx = (self.box[0]+self.box[2])/2
            cy = (self.box[1]+self.box[3])/2
            ranked = sorted(((math.hypot((b[0]+b[2])/2-cx, (b[1]+b[3])/2-cy), i)
                             for i, (b, _, _) in enumerate(detections)))
            if ranked and ranked[0][0] <= 0.15*math.hypot(shape[0], shape[1]):
                chosen = ranked[0][1]
            else:
                self.gap += 1
                if self.gap <= self.max_gap:
                    return None, 'DETECTION GAP - ID ' + str(self.track_id)
                self.reset()
        if self.box is None and detections:
            chosen = max(range(len(detections)), key=lambda i: detections[i][1])
            self.track_id = self.next_id
            self.next_id += 1
        if chosen is None:
            return None, 'SEARCHING'
        self.box = detections[chosen][0]
        self.gap = 0
        self.trail.append((int((self.box[0]+self.box[2])/2), int((self.box[1]+self.box[3])/2)))
        return chosen, 'TRACKING ID ' + str(self.track_id)


def draw_tracking(frame, detections, selected, tracker, state, device_label):
    scale = min(1., 960. / frame.shape[1])
    if scale < 1:
        frame = cv2.resize(frame, (960, round(frame.shape[0] * scale)))
    height, width = frame.shape[:2]

    # The short display trail uses recent measured centres only.
    if (selected is not None or tracker.flow_box is not None) and len(tracker.trail) > 1:
        points = np.asarray(tracker.trail, dtype=np.float32) * scale
        smoothed = np.array([points[max(0, i - 2):i + 1].mean(axis=0)
                             for i in range(len(points))])
        smoothed[-1] = points[-1]
        for i in range(1, len(smoothed)):
            a, b = smoothed[i - 1], smoothed[i]
            if np.linalg.norm(b - a) > 45:
                continue
            strength = .3 + .7 * i / (len(smoothed) - 1)
            color = tuple(int(v * strength) for v in (255, 220, 75))
            cv2.line(frame, tuple(a.round().astype(int)), tuple(b.round().astype(int)),
                     color, 2, cv2.LINE_AA)
        cv2.circle(frame, tuple(points[-1].round().astype(int)), 3,
                   (255, 230, 120), -1, cv2.LINE_AA)

    if selected is None and tracker.flow_box is not None:
        x1,y1,x2,y2=np.round(np.asarray(tracker.flow_box)*scale).astype(int)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(30,185,250),2,cv2.LINE_AA)
        cv2.putText(frame,'FLOW ID '+str(tracker.track_id)+' | no new detection',(max(0,x1),max(40,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.45,(30,185,250),1,cv2.LINE_AA)
    for i, (box, score, class_id) in enumerate(detections):
        # Single-target view: only a temporally confirmed, associated detection
        # gets a box. Raw/unmatched proposals remain in the experiment log.
        if i != selected:
            continue
        coords = np.round(np.asarray(box) * scale).astype(int)
        x1, x2 = np.clip(coords[[0, 2]], 0, width - 1)
        y1, y2 = np.clip(coords[[1, 3]], 0, height - 1)
        color = (80, 235, 115) if i == selected else (60, 190, 245)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (15, 25, 20), 4, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f'drone | model score {score:.2f}'
        if i == selected:
            label = f'ID {tracker.track_id}  |  ' + label
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .48, 1)
        left = max(0, min(x1, width - tw - 14))
        top = max(37, y1 - th - baseline - 12)
        bottom = min(height - 1, top + th + baseline + 12)
        cv2.rectangle(frame, (left, top), (left + tw + 12, bottom), (22, 38, 28), -1)
        cv2.putText(frame, label, (left + 6, top + th + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (245, 255, 245), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, 0), (width, 30), (20, 25, 30), -1)
    cv2.putText(frame, state + ' | ' + device_label, (10, 21),
                cv2.FONT_HERSHEY_SIMPLEX, .48, (245, 245, 245), 1, cv2.LINE_AA)
    return frame


PAGE = b'''<!doctype html><html lang="en"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Drone video preview</title><style>
body{font:16px system-ui;background:#101820;color:#edf2f6;margin:0;padding:20px;max-width:1100px;margin:auto}
h1{font-size:25px;margin-bottom:5px}p{color:#bdcbd5}button{font:inherit;padding:10px 18px;margin:5px 8px 10px 0;border:0;border-radius:6px;cursor:pointer;background:#e4edf4;color:#142535}
img{width:100%;background:#000;border-radius:8px;min-height:150px}#status{white-space:pre-wrap;font-variant-numeric:tabular-nums}</style>
<h1>Drone detection and tracking</h1><p>Recorded RGB video playing as a live feed | __DEVICE__</p>
<button onclick="act('start')">Start / Resume</button><button onclick="act('pause')">Pause</button>
<button onclick="act('restart')">Replay</button><button onclick="act('next')">Next video</button>
<div><button id="record" disabled onclick="act('record-start')">Record and play</button>
<button id="save" disabled onclick="saveClip()">Stop and download MP4</button>
<a id="download" href="/download" style="display:none;color:#8ee4ff">Download last clip</a></div>
<div id="recordStatus" aria-live="polite"></div>
<div id="status">Waiting for the detector...</div><p><img src="/stream" alt="Drone video stream"></p>
<p>Detection plus short-lived optical flow; orange FLOW boxes have no new detection. Drone subtype is unverified; original predicted class IDs remain in logs. Optional small-target views cost additional inference time. Recording uses source playback FPS.</p>
<script>async function act(a){let r=await fetch('/'+a,{method:'POST'});if(!r.ok)throw new Error(await r.text());}
async function saveClip(){try{await act('record-stop');let s=await(await fetch('/status')).json();if(s.download_ready)window.location='/download';else alert('No frames recorded yet.');}catch(e){alert(e.message);}}async function poll(){try{let s=await(await fetch('/status')).json();document.getElementById('record').disabled=s.recording||!!s.error||s.state==='INITIALIZING';document.getElementById('save').disabled=!s.recording;document.getElementById('download').style.display=s.download_ready?'inline':'none';document.getElementById('recordStatus').textContent=s.recording_error||(s.recording?`Recording: ${s.recorded_frames} frames / ${s.recorded_seconds.toFixed(1)} seconds`:s.download_ready?'Recording saved. Ready to download.':'Press Record and play to save a clip.');document.getElementById('status').textContent=s.error||`${s.state} | Video ${s.index+1}/${s.total} | ${s.video}\nPosition ${s.seconds.toFixed(1)} s | Processing ${s.fps.toFixed(1)} FPS | ${s.detections} detections`;}catch(e){}setTimeout(poll,700)}poll()</script></html>'''


class App:
    def __init__(self, videos, record_dir, backend):
        self.videos = videos
        self.record_dir = record_dir
        self.backend = backend
        self.device_label = backend
        self.recording = False
        self.record_writer = None
        self.record_path = None
        self.record_frames = 0
        self.record_fps = 0.
        self.last_recording = None
        self.lock = threading.Condition()
        self.stop = threading.Event()
        self.paused = True
        self.index = 0
        self.generation = 0
        self.jpeg = None
        self.sequence = 0
        self.info = dict(state='INITIALIZING', index=0, total=len(videos),
                         video=videos[0].parent.name, seconds=0., fps=0., detections=0, error='',
                         backend=backend, recording=False, recorded_frames=0, download_ready=False,
                         recording_error='', recorded_seconds=0.)

    def worker(self, detector_context):
        cap = None
        try:
            with detector_context() as predict:
                print(self.backend + ' ready. Open the browser and press Start.', flush=True)
                current = None
                count = 0
                fps = 25.
                tracker = Tracker()
                with self.lock:
                    self.info['state'] = 'READY - press Start'
                while not self.stop.is_set():
                    with self.lock:
                        index, generation = self.index, self.generation
                        paused = self.paused
                    if paused:
                        self.stop.wait(.05)
                        continue
                    tick = time.monotonic()
                    if current != (index, generation):
                        if cap is not None:
                            cap.release()
                        cap = cv2.VideoCapture(str(self.videos[index]))
                        if not cap.isOpened():
                            raise RuntimeError('Cannot open video: '+str(self.videos[index]))
                        fps = float(cap.get(cv2.CAP_PROP_FPS))
                        if not math.isfinite(fps) or fps < 1 or fps > 240:
                            fps = 25.
                        count = 0
                        tracker = Tracker(max_gap=max(1, round(fps*.5)))
                        current = (index, generation)
                    ok, frame = cap.read()
                    if not ok:
                        with self.lock:
                            self.finish_recording()
                            self.paused = True
                            self.info['state'] = 'VIDEO ENDED - Replay or Next video'
                        continue
                    detections = predict(frame)
                    selected, state = tracker.update(detections, frame.shape, frame)
                    frame = draw_tracking(frame, detections, selected, tracker,
                                          state, self.device_label)
                    with self.lock:
                        self.record_frame(frame, fps)
                    encoded_ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not encoded_ok:
                        raise RuntimeError('JPEG encoding failed')
                    count += 1
                    # Include drawing and JPEG encoding in the processing FPS.
                    processing_fps = 1/max(time.monotonic()-tick, .000001)
                    with self.lock:
                        self.jpeg = encoded.tobytes()
                        self.sequence += 1
                        self.info.update(state=state, index=index, video=self.videos[index].parent.name,
                                         seconds=count/fps, fps=processing_fps, detections=len(detections))
                        self.lock.notify_all()
                    # At most source playback speed; no frame skipping.
                    self.stop.wait(max(0, 1/fps-(time.monotonic()-tick)))
        except Exception as exc:
            traceback.print_exc()
            with self.lock:
                self.info.update(state='FAULT', error=str(exc))
                self.lock.notify_all()
        finally:
            if cap is not None:
                cap.release()
            with self.lock:
                self.finish_recording()



    def start_recording(self):
        # Call recording methods while holding self.lock.
        if self.recording:
            return
        if self.info['state'] in ('FAULT', 'INITIALIZING'):
            raise ValueError('Wait until the detector is ready.')
        if self.info['state'].startswith('VIDEO ENDED'):
            self.generation += 1
        self.recording = True
        self.paused = False
        self.record_frames = 0
        self.last_recording = None
        self.info.update(recording=True, recorded_frames=0, recorded_seconds=0., recording_error='', download_ready=False)

    def record_frame(self, frame, fps):
        if not self.recording:
            return
        if self.record_writer is None:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            name = f"{self.backend}_tracking_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
            self.record_path = self.record_dir / (name + '.partial.mp4')
            h, w = frame.shape[:2]
            self.record_writer = cv2.VideoWriter(str(self.record_path),
                                                 cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            self.record_fps = fps
            if not self.record_writer.isOpened():
                self.record_writer.release()
                self.record_writer = None
                self.recording = False
                self.info.update(recording=False, recording_error='Could not create the MP4 recording.')
                return
        self.record_writer.write(frame)
        self.record_frames += 1
        self.info.update(recorded_frames=self.record_frames,
                         recorded_seconds=self.record_frames / self.record_fps)

    def finish_recording(self):
        self.recording = False
        self.info['recording'] = False
        if self.record_writer is None:
            return
        self.record_writer.release()
        self.record_writer = None
        try:
            if self.record_frames:
                finished = self.record_path.with_name(self.record_path.name.replace('.partial.mp4', '.mp4'))
                self.record_path.replace(finished)
                self.last_recording = finished
                self.info['download_ready'] = True
        except OSError as exc:
            self.info['recording_error'] = 'Could not save the recording: ' + str(exc)


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            if self.path == '/':
                content, kind = PAGE.replace(b'__DEVICE__', app.device_label.encode()), 'text/html; charset=utf-8'
            elif self.path == '/status':
                with app.lock:
                    content = json.dumps(dict(app.info)).encode()
                kind = 'application/json'
            elif self.path == '/download':
                with app.lock:
                    recording = app.last_recording
                if recording is None:
                    self.send_error(404, 'Record a clip and stop it first.')
                    return
                try:
                    with recording.open('rb') as video:
                        self.send_response(200)
                        self.send_header('Content-Type', 'video/mp4')
                        self.send_header('Content-Length', str(recording.stat().st_size))
                        self.send_header('Content-Disposition', 'attachment; filename="' + recording.name + '"')
                        self.end_headers()
                        while True:
                            chunk = video.read(256 * 1024)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            elif self.path == '/stream':
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                seen = -1
                try:
                    while not app.stop.is_set():
                        with app.lock:
                            app.lock.wait_for(lambda: app.sequence != seen or app.stop.is_set(), timeout=2)
                            if app.jpeg is None or app.sequence == seen:
                                continue
                            jpeg, seen = app.jpeg, app.sequence
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n')
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass
                return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            with app.lock:
                if self.path == '/start':
                    app.paused = False
                elif self.path == '/pause':
                    app.paused = True
                    app.info['state'] = 'PAUSED'
                elif self.path == '/restart':
                    app.finish_recording()
                    app.generation += 1
                    app.paused = False
                elif self.path == '/next':
                    app.finish_recording()
                    if app.index+1 < len(app.videos):
                        app.index += 1
                        app.generation += 1
                        app.paused = False
                    else:
                        app.paused = True
                        app.info['state'] = 'LAST VIDEO - five-clip limit reached'
                elif self.path == '/record-start':
                    try:
                        app.start_recording()
                    except ValueError as exc:
                        self.send_error(409, str(exc))
                        return
                elif self.path == '/record-stop':
                    app.finish_recording()
                else:
                    self.send_error(404)
                    return
            self.send_response(204)
            self.end_headers()
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('hailo', 'pc'), default='hailo')
    parser.add_argument('--hef', type=Path)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--model-name', default='HusAero RGB5')
    parser.add_argument('--small-target',action='store_true',help='Try four overlapping views after a miss; increases inference cost')
    parser.add_argument('--manifest', type=Path, help='HEF deployment.json with names and hef_sha256')
    parser.add_argument('--video-root', type=Path, default=VIDEO_ROOT)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--record-dir', type=Path, default=ROOT/'results/recordings')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host')
    args = parser.parse_args()
    if args.backend == 'pc' and args.weights is None:
        parser.error('--weights is required for --backend pc')
    if args.backend == 'hailo' and args.hef is None:
        parser.error('--hef is required for --backend hailo')
    model_path = args.weights if args.backend == 'pc' else args.hef
    if args.backend == 'hailo':
        if args.manifest is None:
            parser.error('--manifest is required for the RGB5 Hailo model')
        metadata = json.loads(args.manifest.read_text())
        if metadata.get('names') != NAMES:
            parser.error('Deployment class map does not match RGB5')
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata.get('hef_sha256'):
            parser.error('HEF hash does not match the deployment manifest')
        if args.model_name == 'HusAero RGB5':
            args.model_name = metadata.get('model', args.model_name)
    videos = [args.video] if args.video else [args.video_root/name/'visible.mp4' for name in VIDEO_NAMES]
    missing = [str(p) for p in [model_path, *videos] if not p.is_file()]
    if missing:
        raise SystemExit('Missing files: ' + ', '.join(missing))
    backend = ('Windows PC / CPU' if args.backend == 'pc' else 'Raspberry Pi / Hailo-8') + ' / ' + args.model_name
    app = App(videos, args.record_dir, args.backend)
    app.device_label = backend
    app.info['backend'] = backend
    app.info['model_name'] = args.model_name
    app.info['model_sha256'] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    # Filenames use the simple backend name rather than the display label.
    host = args.host or ('127.0.0.1' if args.backend == 'pc' else '0.0.0.0')
    server = ThreadingHTTPServer((host, args.port), make_handler(app))
    factory = partial(detector, args.backend, model_path, args.small_target)
    worker = threading.Thread(target=app.worker, args=(factory,), daemon=True)
    worker.start()
    print(f'Browser: http://localhost:{args.port} | Ctrl+C to stop', flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        print('Stopping...', flush=True)
    finally:
        app.stop.set()
        with app.lock:
            app.lock.notify_all()
        server.server_close()
        worker.join()


if __name__ == '__main__':
    main()
