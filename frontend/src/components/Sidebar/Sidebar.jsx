import React, { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './Sidebar.css';

const VIEW_MODES = [
  { id: 'feature',  icon: '🔬', label: 'Feature Maps',           desc: 'Xem feature maps qua từng layer' },
  { id: 'gradient', icon: '📈', label: 'Gradient Analysis',      desc: 'Phân tích gradient qua mạng' },
  { id: 'heatmap',  icon: '🔥', label: 'Grad-CAM Heatmap',       desc: 'Vùng ảnh hưởng đến kết quả' },
];

export default function Sidebar({ onUpload, result, viewMode, onViewModeChange, status, modelType, onModelTypeChange }) {
  const [isOpen, setIsOpen] = useState(true);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setPreviewUrl(URL.createObjectURL(file));
    onUpload(file);
  }, [onUpload]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setPreviewUrl(URL.createObjectURL(file));
      onUpload(file);
    }
  }, [onUpload]);

  const handleClear = useCallback(() => {
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, []);

  return (
    <>
      <motion.aside 
        initial={false}
        animate={{ width: isOpen ? 320 : 64 }}
        className={`sidebar ${!isOpen ? 'sidebar--collapsed' : ''}`}
        id="sidebar"
      >
        <div className="sidebar__header">
          {isOpen && <motion.div initial={{opacity:0}} animate={{opacity:1}} className="sidebar__title">Menu Hệ Thống</motion.div>}
          <button 
            className="sidebar__toggle" 
            onClick={() => setIsOpen(!isOpen)}
            title={isOpen ? "Thu nhỏ" : "Mở rộng"}
          >
            {isOpen ? '◀' : '▶'}
          </button>
        </div>

        <div className="sidebar__body" style={{ overflow: isOpen ? 'auto' : 'hidden' }}>
          {/* Model Selection */}
          <div className="sidebar__section">
            <div className="sidebar__section-label">
              {isOpen ? '🧠 Chọn Mô hình' : '🧠'}
            </div>
            {isOpen ? (
              <div className="model-selector">
                <label className="model-selector__option">
                  <input type="radio" name="modelType" value="imagenet" checked={modelType === 'imagenet'} onChange={() => onModelTypeChange('imagenet')} />
                  <span>Vạn vật (ImageNet)</span>
                </label>
                <label className="model-selector__option">
                  <input type="radio" name="modelType" value="catsdogs" checked={modelType === 'catsdogs'} onChange={() => onModelTypeChange('catsdogs')} />
                  <span>Chó Mèo (Custom)</span>
                </label>
              </div>
            ) : (
              <div className="sidebar__mini-icon" onClick={() => setIsOpen(true)}>🧠</div>
            )}
          </div>

          {/* Upload Section */}
          <div className="sidebar__section">
            <div className="sidebar__section-label">
              {isOpen ? '📤 Upload Ảnh' : '📤'}
            </div>
            {isOpen ? (
              !previewUrl ? (
                <div
                  className={`upload-zone ${isDragging ? 'upload-zone--dragging' : ''}`}
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                >
                  <div className="upload-zone__icon">🖼️</div>
                  <div className="upload-zone__text">Kéo thả ảnh vào đây</div>
                  <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileChange} hidden />
                </div>
              ) : (
                <div className="upload-preview">
                  <img src={previewUrl} alt="Preview" />
                  <button className="upload-preview__btn" onClick={handleClear}>✕</button>
                </div>
              )
            ) : (
               <div className="sidebar__mini-icon" onClick={() => setIsOpen(true)}>🖼️</div>
            )}
          </div>

          {/* Classification Result */}
          {result && isOpen && (
            <motion.div initial={{opacity:0, y:10}} animate={{opacity:1, y:0}} className="sidebar__section">
              <div className="sidebar__section-label">🎯 Kết quả</div>
              <div className="prediction-card">
                <div className="prediction-card__label">{result.prediction}</div>
                <div className="prediction-card__confidence">{(result.confidence * 100).toFixed(1)}%</div>
              </div>
            </motion.div>
          )}

          {/* View Modes */}
          <div className="sidebar__section">
            <div className="sidebar__section-label">
              {isOpen ? '👁️ Chế độ xem' : '👁️'}
            </div>
            <div className={`view-modes ${!isOpen ? 'view-modes--mini' : ''}`}>
              {VIEW_MODES.map(mode => (
                <button
                  key={mode.id}
                  className={`view-mode-btn ${viewMode === mode.id ? 'view-mode-btn--active' : ''}`}
                  onClick={() => {
                    onViewModeChange(mode.id);
                    if (!isOpen) setIsOpen(true);
                  }}
                  title={mode.label}
                >
                  <span className="view-mode-btn__icon">{mode.icon}</span>
                  {isOpen && <span className="view-mode-btn__text">{mode.label}</span>}
                </button>
              ))}
            </div>
          </div>
        </div>

        {isOpen && (
          <div className="sidebar__footer">
            ResNet50 v1.0
          </div>
        )}
      </motion.aside>
    </>
  );
}
