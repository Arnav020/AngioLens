export default function Disclaimer({ onClose }: { onClose: () => void }) {
  return (
    <div className="modal-bg" role="dialog" aria-modal="true" aria-labelledby="disc-title">
      <div className="modal">
        <div className="eyebrow">Clinical safety notice</div>
        <h2 id="disc-title">This is a virtual angiogram, not a real one.</h2>
        <p>
          AngioLens estimates the probability of coronary artery disease and of ≥50% stenosis in the LAD, LCX and RCA
          from routine clinical, ECG, laboratory and echo data. It was trained on 303 patients from a single centre.
        </p>
        <ul>
          <li>For <b>decision support and education only</b>. It is not a diagnosis and not a medical device.</li>
          <li>It does <b>not</b> replace invasive coronary angiography, CT coronary angiography or clinical judgement.</li>
          <li>The 3D heart shows a <b>generic anatomy</b>. Colours map model probabilities onto standard vessel
            territories, not onto this patient's imaged lesions.</li>
          <li>Feature attributions are statistical associations, not causes.</li>
        </ul>
        <button className="btn primary" onClick={onClose} autoFocus>
          I understand, continue
        </button>
      </div>
    </div>
  );
}
