/**
 * App.jsx
 * =======
 * Component gốc – quản lý state toàn cục và bố cục 3 cột.
 *   Sidebar (trái)  |  Main Canvas (giữa)  |  Detail Panel (phải)
 */

import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './App.css';

import Sidebar from './components/Sidebar/Sidebar';
import PipelineFlow from './components/PipelineFlow/PipelineFlow';
import DetailPanel from './components/DetailPanel/DetailPanel';

const API_BASE = 'http://localhost:8000';

export default function App() {
  // ---- State ----
  const [result, setResult] = useState(null);          // JSON trả từ /predict
  const [selectedLayer, setSelectedLayer] = useState(null); // Layer đang chọn
  const [viewMode, setViewMode] = useState('feature');  // feature | gradient | heatmap
  const [status, setStatus] = useState('idle');         // idle | loading | animating | done | error
  const [errorMsg, setErrorMsg] = useState('');
  const [modelType, setModelType] = useState('imagenet'); // imagenet | catsdogs

  // ---- Handle upload ảnh ----
  const handleUpload = useCallback(async (file) => {
    if (!file) return;

    setStatus('loading');
    setResult(null);
    setSelectedLayer(null);
    setErrorMsg('');

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('model_type', modelType);

      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || `Server lỗi ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
      
      // Chuyển sang trạng thái animating để người dùng thấy luồng xử lý
      setStatus('animating');
      setTimeout(() => {
        setStatus('done');
      }, 12000); // Kéo dài thời gian xem animation điện ảnh ngang (12s)

    } catch (err) {
      console.error('Upload error:', err);
      setErrorMsg(err.message || 'Không thể kết nối tới server.');
      setStatus('error');
    }
  }, [modelType]);

  // ---- Handle chọn layer ----
  const handleSelectLayer = useCallback((layerName) => {
    setSelectedLayer(layerName);
  }, []);

  // ---- Lấy dữ liệu layer đang chọn ----
  const selectedLayerData = result?.layers?.find(l => l.name === selectedLayer) || null;

  // ---- Render ----
  return (
    <div className="app">
      <Sidebar
        onUpload={handleUpload}
        result={result}
        viewMode={viewMode}
        onViewModeChange={setViewMode}
        status={status}
        modelType={modelType}
        onModelTypeChange={setModelType}
      />

      <main className="app__main">
        <header className="app__header">
          <div className="app__logo">
            <h1 className="app__logo-text">ResNet50 <span className="gradient-text">Studio</span></h1>
          </div>
          <div className={`app__status-badge app__status-badge--${status}`}>
             {status === 'idle' && 'TRẠNG THÁI CHỜ'}
             {status === 'loading' && 'ĐANG PHÂN TÍCH...'}
             {status === 'animating' && 'CHẠY FLOW ĐIỆN ẢNH...'}
             {status === 'done' && `HOÀN TẤT: ${result?.prediction}`}
             {status === 'error' && 'LỖI'}
          </div>
        </header>

        <div className="app__content">
           <AnimatePresence mode="wait">
              {!result && status === 'idle' ? (
                <motion.div 
                   key="empty" 
                   initial={{opacity:0, scale:0.95}} animate={{opacity:1, scale:1}} exit={{opacity:0}}
                   className="landing-placeholder"
                >
                   <div className="landing-placeholder__icon">🧬</div>
                   <h2>Khởi tạo Phân tích Deep Learning</h2>
                   <p>Mô hình ResNet50 đã sẵn sàng. Vui lòng tải lên một hình ảnh từ thanh Toolbar bên trái để bắt đầu trích xuất đặc trưng (Feature Extraction).</p>
                   <div className="landing-placeholder__tip">💡 Mẹo: Bạn có thể thu nhỏ Sidebar để mở rộng không gian làm việc.</div>
                </motion.div>
              ) : (
                <motion.div key="workspace" initial={{opacity:0}} animate={{opacity:1}} className="app__workspace">
                    <div className="app__pipeline-area">
                      {(status === 'done' || status === 'animating') && result && (
                        <PipelineFlow
                          result={result}
                          activeNode={status === 'animating' ? null : selectedLayer}
                          onSelectNode={handleSelectLayer}
                          viewMode={status === 'animating' ? 'feature' : viewMode}
                          isAnimating={status === 'animating'}
                        />
                      )}
                      
                      {status === 'loading' && (
                        <div className="app__loading-overlay">
                           <div className="app__spinner" />
                           <p>Đang chạy Forward Propagation qua 50 layers...</p>
                        </div>
                      )}

                      {status === 'error' && (
                        <div className="empty-state">
                           <div className="empty-state__icon">⚠️</div>
                           <h2 className="empty-state__title">Có lỗi xảy ra</h2>
                           <p className="empty-state__desc">{errorMsg}</p>
                        </div>
                      )}
                    </div>

                    {/* Pixel-Level Viewer là một phần của DetailPanel */}
                    <DetailPanel
                      layerData={selectedLayerData}
                      viewMode={viewMode}
                      gradcamOverlay={result?.gradcam_overlay}
                      gradcamHeatmap={result?.gradcam_heatmap}
                      result={result}
                    />
                </motion.div>
              )}
           </AnimatePresence>
        </div>
      </main>
    </div>
  );
}
