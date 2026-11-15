"""Train HusAero-8n then HusAero-11n on RGB4. Validation only; NO TEST evaluation."""
import argparse
import gc
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

CLASSES = ['drone', 'bird', 'airplane', 'helicopter']


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    repo = args.repo.resolve()
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    import torch
    import yaml
    import ultralytics
    from ultralytics import YOLO
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable. Training stopped; CPU fallback is disabled.')

    candidates = ([args.preflight / 'preflight_report.json'] if args.preflight else
                  sorted((repo / 'results').glob('RGB4_Preflight_*/preflight_report.json'), reverse=True))
    report = None
    for candidate in candidates:
        value = json.loads(candidate.read_text(encoding='utf-8'))
        if value.get('status') == 'PASS' and value.get('full_dataset_check') is True and value.get('classes') == CLASSES:
            report, report_path = value, candidate
            break
    if report is None:
        raise RuntimeError('No successful full RGB4 preflight found. Run preflight_rgb4.py first.')
    data = repo / 'dataset/combined_rgb4/data.yaml'
    definition = yaml.safe_load(data.read_text(encoding='utf-8'))
    assert definition['names'] == CLASSES, 'Dataset class order changed'
    for split in ['train', 'val']:
        image_path = Path(definition[split])
        if not image_path.is_absolute():
            image_path = data.parent / image_path
        assert image_path.is_dir(), f'Missing {split} images: {image_path}'
    initialized = []
    for name in ['HusAero-8n', 'HusAero-11n']:
        item = next(v for v in report['models'] if v['model'] == name)
        checkpoint = report_path.parent / Path(item['initialized']).name
        assert sha(checkpoint) == item['initialized_sha256'], f'Initialized weights changed: {name}'
        source = repo / 'models/rgb5_v2' / name / (name + '_RGB5.pt')
        assert sha(source) == item['source_sha256'], f'Original weights changed: {name}'
        model = YOLO(str(checkpoint))
        assert list(model.names.values()) == CLASSES and model.model.model[-1].nc == 4
        assert model.ckpt.get('epoch') == -1 and model.ckpt.get('optimizer') is None
        del model
        initialized.append((name, checkpoint))

    print('READY: 8n -> 11n | 100 epochs maximum EACH | patience 30 | batch 32', flush=True)
    print('GPU: ' + torch.cuda.get_device_name(0), flush=True)
    print('Four classes: ' + ', '.join(CLASSES), flush=True)
    print('Validation during training. NO TEST evaluation. Originals preserved.', flush=True)
    if args.check_only:
        print('CHECK PASS - NO TRAINING STARTED', flush=True)
        return

    output = repo / 'results' / ('RGB4_Training_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(epochs=100, patience=30, imgsz=640, batch=32, device=0,
                    workers=4, amp=True, cache=False, seed=42, deterministic=True,
                    optimizer='SGD', lr0=0.001, resume=False, val=True,
                    save=True, plots=True, exist_ok=False)
    state = dict(status='running', started_at=datetime.now().isoformat(),
                 preflight=str(report_path), data=str(data), classes=CLASSES,
                 ultralytics=ultralytics.__version__, torch=torch.__version__,
                 gpu=torch.cuda.get_device_name(0), settings=settings,
                 test_evaluation=False, models=[dict(name=n, status='waiting',
                 starting_weights=str(p), epochs_completed=0) for n, p in initialized])
    write_json(output / 'progress.json', state)
    print('Results: ' + str(output), flush=True)
    try:
        for index, (name, checkpoint) in enumerate(initialized):
            entry = state['models'][index]
            entry['status'] = 'training'
            print(f'\n========== MODEL {index + 1}/2: {name} ==========\n', flush=True)
            write_json(output / 'progress.json', state)
            model = YOLO(str(checkpoint))

            def epoch_finished(trainer):
                metrics = {str(k): float(v) for k, v in trainer.metrics.items()}
                entry.update(epochs_completed=trainer.epoch + 1, validation=metrics,
                             best_validation_fitness=float(trainer.best_fitness),
                             updated_at=datetime.now().isoformat())
                write_json(output / 'progress.json', state)
                map95 = metrics.get('metrics/mAP50-95(B)')
                score = f'{map95:.4f}' if map95 is not None else 'see results.csv'
                print(f'\n[MODEL {index + 1}/2 | {name}] EPOCH {trainer.epoch + 1}/100 '
                      f'| validation mAP50-95: {score}\n', flush=True)

            model.add_callback('on_fit_epoch_end', epoch_finished)
            model.train(data=str(data), project=str(output), name=name, **settings)
            run_dir = Path(model.trainer.save_dir)
            best = run_dir / 'weights/best.pt'
            assert best.is_file(), f'Best checkpoint missing: {best}'
            entry.update(status='completed', best_weights=str(best), best_sha256=sha(best),
                         last_weights=str(run_dir / 'weights/last.pt'), run_dir=str(run_dir))
            write_json(output / 'progress.json', state)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        state['status'] = 'completed'
        state['finished_at'] = datetime.now().isoformat()
        write_json(output / 'progress.json', state)
        print('\nBOTH MODELS FINISHED. NO TEST EVALUATION WAS RUN.', flush=True)
        for entry in state['models']:
            print(entry['name'] + ' best: ' + entry['best_weights'], flush=True)
        print('These .pt files need export/compilation for Hailo-8 before Pi testing.', flush=True)
    except BaseException as error:
        state['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        state['error'] = str(error)
        for entry in state['models']:
            if entry['status'] == 'training':
                entry['status'] = state['status']
        write_json(output / 'progress.json', state)
        raise


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
