export default function SquircleDefs() {
  return (
    <svg width="0" height="0" style={{ position: 'absolute' }}>
      <defs>
        <clipPath id="squircle" clipPathUnits="objectBoundingBox">
          <path d="M 0,0.5 C 0,0.1 0.1,0 0.5,0 C 0.9,0 1,0.1 1,0.5 C 1,0.9 0.9,1 0.5,1 C 0.1,1 0,0.9 0,0.5 Z" />
        </clipPath>
      </defs>
    </svg>
  );
}
