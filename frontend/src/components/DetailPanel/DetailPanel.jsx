/**
 * DetailPanel.jsx
 * ===============
 * Panel bên phải – hiển thị thông tin chi tiết cho layer đang được chọn.
 *   - Tên, loại, kernel, output shape
 *   - Vai trò trong mạng
 *   - Gradient magnitude
 *   - Grad-CAM preview (nếu ở heatmap mode)
 */

import './DetailPanel.css';
import LayerTraversalViewer from './LayerTraversalViewer';

// Map icon theo layer type
const TYPE_ICONS = {
  'Conv2D':           '🔲',
  'BatchNorm':        '📐',
  'ReLU':             '⚡',
  'MaxPool2D':        '📉',
  'ResNet Stage':     '🧱',
  'AdaptiveAvgPool2D':'🎯',
};

export default function DetailPanel({ layerData, viewMode, gradcamOverlay, gradcamHeatmap, result }) {
  if (!layerData) {
    return (
       <div className="p-4 bg-gray-900 min-h-screen text-white">
         <div className="text-gray-400 italic mb-4">Xin mời chọn một Layer bên cạnh, hoặc khám phá Pixel-level bên dưới:</div>
         <LayerTraversalViewer result={result} />
       </div>
    );
  }

  const icon = TYPE_ICONS[layerData.type] || '📦';
  const shape = layerData.output_shape;
  const shapeStr = shape
    ? (shape.length === 4
      ? `${shape[1]} × ${shape[2]} × ${shape[3]}`
      : shape.join(' × '))
    : '—';

  return (
    <aside className="detail-panel" id="detail-panel">
      <div className="detail-panel__header">
        <span className="detail-panel__title">Chi tiết Layer</span>
      </div>

      <div className="detail-panel__body">
        {/* Layer name */}
        <div className="detail-layer-name animate-fade-in">
          <div className="detail-layer-name__icon">{icon}</div>
          <div className="detail-layer-name__text">
            <h2>{layerData.name}</h2>
            <span>{layerData.type}</span>
          </div>
        </div>

        {/* Info cards */}
        <div className="detail-info-grid animate-fade-in" style={{ animationDelay: '0.05s' }}>
          <div className="detail-info-card">
            <div className="detail-info-card__label">Kernel Size</div>
            <div className="detail-info-card__value">{layerData.kernel}</div>
          </div>
          <div className="detail-info-card">
            <div className="detail-info-card__label">Stride</div>
            <div className="detail-info-card__value">{layerData.stride}</div>
          </div>
          <div className="detail-info-card">
            <div className="detail-info-card__label">Filters</div>
            <div className="detail-info-card__value">{layerData.filters}</div>
          </div>
          <div className="detail-info-card">
            <div className="detail-info-card__label">Output Shape</div>
            <div className="detail-info-card__value detail-info-card__value--small">
              {shapeStr}
            </div>
          </div>
        </div>

        {/* Role description */}
        {layerData.role && (
          <div className="detail-role animate-fade-in" style={{ animationDelay: '0.1s' }}>
            <div className="detail-role__label">💡 Vai trò trong mạng</div>
            <div className="detail-role__text">{layerData.role}</div>
          </div>
        )}

        {/* Gradient magnitude */}
        {layerData.gradient_mag !== undefined && (
          <div className="detail-gradient animate-fade-in" style={{ animationDelay: '0.15s' }}>
            <div className="detail-gradient__label">📈 Gradient Magnitude</div>
            <div className="detail-gradient__value" style={{
              color: layerData.gradient_mag > 0.01
                ? 'var(--warning)'
                : layerData.gradient_mag > 0.001
                  ? 'var(--info)'
                  : 'var(--success)'
            }}>
              {layerData.gradient_mag.toExponential(3)}
            </div>
            <div className="detail-gradient__bar">
              <div
                className="detail-gradient__bar-fill"
                style={{ width: `${Math.min(layerData.gradient_mag * 1000, 100)}%` }}
              />
            </div>
          </div>
        )}

        {/* Feature maps count */}
        {layerData.feature_maps && layerData.feature_maps.length > 0 && (
          <div className="detail-features-count animate-fade-in" style={{ animationDelay: '0.2s' }}>
            🔬 Đang hiển thị <strong>{layerData.feature_maps.length}</strong> / {layerData.filters} feature maps
          </div>
        )}

        {/* Grad-CAM (chỉ hiện khi viewMode === heatmap) */}
        {viewMode === 'heatmap' && (gradcamOverlay || gradcamHeatmap) && (
          <div className="detail-gradcam animate-fade-in" style={{ animationDelay: '0.2s' }}>
            <div className="detail-gradcam__label">🔥 Grad-CAM</div>
            <div className="detail-gradcam__images">
              {gradcamOverlay && (
                <div className="detail-gradcam__card">
                  <img
                    src={`data:image/png;base64,${gradcamOverlay}`}
                    alt="Grad-CAM overlay"
                  />
                  <div className="detail-gradcam__card-label">Overlay</div>
                </div>
              )}
              {gradcamHeatmap && (
                <div className="detail-gradcam__card">
                  <img
                    src={`data:image/png;base64,${gradcamHeatmap}`}
                    alt="Heatmap"
                  />
                  <div className="detail-gradcam__card-label">Heatmap</div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
