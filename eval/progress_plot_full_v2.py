import numpy as np
import cv2

from data.utils import vx_vy2v_theta


def _flow_to_bgr(flow, vmax = 256.):
    """Convert an optical-flow sequence [n_frames, 2, H, W] to Lab-colorwheel BGR frames."""
    n_frames, _, H, W = flow.shape
    Lab = np.zeros((n_frames, H, W, 3))
    vsincos = vx_vy2v_theta(flow)
    saturated = vsincos[:, 0, :, :] > vmax
    Lab[:, :, :, 0] = 100 * vsincos[:, 0, :, :] / vmax
    Lab[:, :, :, 1] = 127 * vsincos[:, 2, :, :]
    Lab[:, :, :, 2] = 127 * vsincos[:, 1, :, :]
    Lab[saturated] = [[100, 0, 0]]

    bgr = np.zeros((n_frames, H, W, 3), dtype = np.uint8)
    for f in range(n_frames):
        bgr[f] = (cv2.cvtColor(Lab[f].astype(np.float32), cv2.COLOR_LAB2BGR) * 255).astype(np.uint8)
    return bgr, Lab


def _events_to_bgr(events, vmax = None):
    """Convert an ON/OFF event-count sequence [n_frames, 2, H, W] to a red/blue-on-white BGR preview.

    Channel 0 is ON (positive polarity) counts, channel 1 is OFF (negative polarity)
    counts (matching dsec_dataset_lite/data/event2frame.py:cumulate_spikes_into_frames).
    ON pixels are pushed towards red, OFF pixels towards blue, on a white background;
    pixels with no events in the window stay white.
    """
    n_frames, _, H, W = events.shape
    on = events[:, 0].astype(np.float32)
    off = events[:, 1].astype(np.float32)

    if vmax is None:
        nonzero = np.concatenate([on[on > 0], off[off > 0]])
        vmax = np.percentile(nonzero, 99) if nonzero.size else 1.0
        vmax = max(float(vmax), 1.0)

    on_norm = np.clip(on / vmax, 0., 1.)
    off_norm = np.clip(off / vmax, 0., 1.)

    bgr = np.full((n_frames, H, W, 3), 255., dtype = np.float32)
    bgr[..., 0] -= on_norm * 255.   # ON  -> reduce B, G (towards red)
    bgr[..., 1] -= on_norm * 255.
    bgr[..., 1] -= off_norm * 255.  # OFF -> reduce G, R (towards blue)
    bgr[..., 2] -= off_norm * 255.

    return np.clip(bgr, 0, 255).astype(np.uint8)


def _write_panel_video(panel_seqs, panel_labels, fps, filename, label_height = 40):
    """Write a row of labeled BGR panel sequences (each [n_frames, H, W, 3]) to an mp4."""
    n_frames, H, W = panel_seqs[0].shape[:3]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (W * len(panel_seqs), H + label_height), isColor = True)

    for f in range(n_frames):
        row = []
        for seq, label in zip(panel_seqs, panel_labels):
            header = np.zeros((label_height, W, 3), dtype = np.uint8)
            cv2.putText(header, label, (10, label_height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            row.append(np.concatenate([header, seq[f]], axis = 0))
        out.write(np.concatenate(row, axis = 1))

    out.release()


def plot_gt_pred_events_camera(gt, pred, events, camera, mask, fps, filename = 'preview.mp4', label_height = 40):
    """Render a 4-panel preview: Ground Truth (masked) | SNN Prediction (masked) | Input Events | Camera.

    Both flow panels are masked using the real per-pixel ground-truth validity
    mask (as loaded from DSECDatasetLite's mask_tensors) rather than
    `plot_evolution`'s "flow happens to be exactly zero" heuristic, so invalid
    pixels are reliably blacked out on both. `camera` is the real rectified
    camera-frame sequence, at its native resolution; it is resized to match the
    flow/event panels for display.

    Parameters
    ----------
    gt, pred : ndarray [n_frames, 2, H, W]
    events : ndarray [n_frames, 2, H, W] -- ON/OFF event counts for the same frames.
    camera : ndarray [n_frames, H_cam, W_cam, 3] -- BGR camera frames (any resolution).
    mask : ndarray [n_frames, H, W] -- ground-truth validity mask (nonzero = valid).
    """
    gt_bgr, _ = _flow_to_bgr(gt)
    pred_bgr, _ = _flow_to_bgr(pred)
    events_bgr = _events_to_bgr(events)

    invalid = np.asarray(mask) == 0
    gt_bgr = gt_bgr.copy()
    pred_bgr = pred_bgr.copy()
    gt_bgr[invalid] = [0, 0, 0]
    pred_bgr[invalid] = [0, 0, 0]

    n_frames, H, W = gt_bgr.shape[:3]
    camera_resized = np.zeros((n_frames, H, W, 3), dtype = np.uint8)
    for f in range(n_frames):
        camera_resized[f] = cv2.resize(camera[f], (W, H))

    _write_panel_video([gt_bgr, pred_bgr, events_bgr, camera_resized],
                       ["Ground Truth (Masked)", "SNN Prediction (Masked)", "Input Events", "Camera"],
                       fps, filename, label_height)


def plot_gt_vs_predictions(gt, predictions, labels, fps, filename = 'comparison.mp4', label_height = 40):
    """Render ground truth alongside one or more masked predictions, side by side.

    Unlike `plot_evolution` (which renders a full gt/pred/error/scale grid per
    prediction), this produces a single row -- [Ground Truth, prediction_1, ...] --
    each prediction masked to the ground-truth's valid region. Useful for comparing
    multiple predictions (e.g. clean vs. attacked) against one shared ground-truth
    panel instead of duplicating it once per prediction.

    Parameters
    ----------
    gt : ndarray [n_frames, 2, H, W]
    predictions : list of ndarray [n_frames, 2, H, W]
    labels : list of str, one per prediction (e.g. ["No attack", "Retiming attack"])
    """
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length.")

    gt_bgr, gt_Lab = _flow_to_bgr(gt)
    useless = gt_Lab[:, :, :, 0] == 0.  # pixels with no ground-truth flow

    panel_seqs = [gt_bgr]
    panel_labels = ["Ground Truth"] + list(labels)

    for pred in predictions:
        pred_bgr, _ = _flow_to_bgr(pred)
        pred_bgr = pred_bgr.copy()
        pred_bgr[useless] = [0, 0, 0]
        panel_seqs.append(pred_bgr)

    n_frames, H, W = gt_bgr.shape[:3]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (W * len(panel_seqs), H + label_height), isColor = True)

    for f in range(n_frames):
        row = []
        for seq, label in zip(panel_seqs, panel_labels):
            header = np.zeros((label_height, W, 3), dtype = np.uint8)
            cv2.putText(header, label, (10, label_height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            row.append(np.concatenate([header, seq[f]], axis = 0))
        out.write(np.concatenate(row, axis = 1))

    out.release()


def plot_evolution(gt, pred, mask, fps, filename = 'comparison.mp4', scale_frame = False):

    n_frames = gt.shape[0]
    H = gt.shape[2]; W = gt.shape[3]

    ## Ground-truth frame generation

    gt_Lab = np.zeros((n_frames, H, W, 3))
    gt_vsincos = vx_vy2v_theta(gt)
    #vmax = np.max(gt_vsincos[:,0,:,:])
    vmax = 256. # A PRIORI FACTOR
    mask_gt = gt_vsincos[:,0,:,:] > vmax

    gt_Lab[:,:,:,0] = 100 * gt_vsincos[:,0,:,:] / vmax
    gt_Lab[:,:,:,1] = 127 * gt_vsincos[:,2,:,:]
    gt_Lab[:,:,:,2] = 127 * gt_vsincos[:,1,:,:]

    gt_Lab[mask_gt] = [[100, 0, 0]]


    ## Pred frame generation

    pred_Lab = np.zeros((n_frames, H, W, 3))
    pred_vsincos = vx_vy2v_theta(pred)
    mask_pred = pred_vsincos[:,0,:,:] > vmax

    pred_Lab[:,:,:,0] = 100 * pred_vsincos[:,0,:,:] / vmax
    pred_Lab[:,:,:,1] = 127 * pred_vsincos[:,2,:,:]
    pred_Lab[:,:,:,2] = 127 * pred_vsincos[:,1,:,:]

    pred_Lab[mask_pred] = [[100, 0, 0]] # White pixels -> saturate pixels with greater value than the maximum gt

    ## Error frame generation

    eps_x = gt[:, 0, :, :] - pred[:, 0, :, :]
    eps_y = gt[:, 1, :, :] - pred[:, 1, :, :]
    eps = np.sqrt(eps_x**2 + eps_y**2)
    eps_max = np.max(eps)
    #print(eps_max)
    #raise

    eps_frames = (255 * eps/eps_max).astype(np.uint8)
    
    ## Scale generation

    L = np.ones((480, 480)) * 100

    a = np.ones((480, 480))
    b = np.ones((480, 480))

    min_value = -127

    for elem in range(480):
        a[:, elem] *= (min_value + elem*(127*2)/479)
        b[elem, :] *= (min_value + elem*(127*2)/479)

    scale_Lab = np.zeros((480, 480, 3))
    scale_Lab[:, :, 0] = L * np.sqrt((a/127)**2 + (b/127)**2)
    scale_Lab[:, :, 1] = a
    scale_Lab[:, :, 2] = b

    outer_circle = np.sqrt(a**2 + b**2) > 127

    #scale_Lab[outer_circle] *= 0 # Black exterior
    scale_Lab[outer_circle] = [[100, 0, 0]]
    
    
    scale_BGR = cv2.cvtColor(scale_Lab.astype(np.float32), cv2.COLOR_LAB2BGR)

    scale_frame = np.ones((H, W, 3))
    scale_frame[:, 80:560, :] = scale_BGR
    scale_frame = (scale_frame * 255).astype(np.uint8)

    ## Video generation
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (3 * W, 2 * H), isColor = True)
    #out = cv2.VideoWriter(filename, fourcc, 1, (2 * W, 2 * H), isColor = True)

    #### NEW PART: MASK PREDICTIONS
    useless_pixels = gt_Lab[:,:,:,0] == 0. # GT greater than 2% of max value
    
 
    masked_pred_Lab = np.copy(pred_Lab)
    masked_pred_Lab[useless_pixels] = [[0, 0, 0]]
    
    eps_frames_masked = np.copy(eps_frames)

    eps_frames_masked[useless_pixels] = 0
    #### NEW PART: MASk PREDICTIONS

    for f in range(n_frames):
        
        gt_BGR = (cv2.cvtColor(gt_Lab[f].astype(np.float32), cv2.COLOR_LAB2BGR) * 255).astype(np.uint8)
        pred_BGR = (cv2.cvtColor(pred_Lab[f].astype(np.float32), cv2.COLOR_LAB2BGR) * 255).astype(np.uint8)
        masked_pred_BGR = (cv2.cvtColor(masked_pred_Lab[f].astype(np.float32), cv2.COLOR_LAB2BGR) * 255).astype(np.uint8)

        eps_grayscale = (cv2.cvtColor(eps_frames[f], cv2.COLOR_GRAY2BGR)*255).astype(np.uint8)
        eps_colormap = (cv2.applyColorMap(eps_grayscale, cv2.COLORMAP_JET)*255).astype(np.uint8)
        
        eps_grayscale_masked = (cv2.cvtColor(eps_frames_masked[f], cv2.COLOR_GRAY2BGR)*255).astype(np.uint8)
        eps_colormap_masked = (cv2.applyColorMap(eps_grayscale_masked, cv2.COLORMAP_JET)*255).astype(np.uint8)
        
        frame_gt_pred = np.concatenate((gt_BGR, masked_pred_BGR, pred_BGR), axis = 1)
        frame_eps_scale = np.concatenate((scale_frame, eps_colormap_masked, eps_colormap), axis = 1)

        frame = np.concatenate((frame_gt_pred, frame_eps_scale), axis = 0)

        out.write(frame)
 
    out.release()
 
