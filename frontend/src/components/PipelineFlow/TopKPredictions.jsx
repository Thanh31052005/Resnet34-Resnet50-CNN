import React from 'react';
import { motion } from 'framer-motion';
import './TopKPredictions.css';

/**
 * TopKPredictions.jsx
 * ==================
 * Trực quan hóa danh sách 5 dự đoán hàng đầu sau lớp Softmax.
 * Hiển thị dưới dạng các thanh xác suất chuyển động mượt mà.
 */
export default function TopKPredictions({ topK }) {
  if (!topK || topK.length === 0) return null;

  return (
    <div className="topk-dashboard">
      <div className="topk-dashboard__header">
         <div className="topk-dashboard__title">🎯 TOP-5 DỰ ĐOÁN XÁC SUẤT</div>
         <div className="topk-dashboard__badge">Softmax Output</div>
      </div>

      <div className="topk-list">
        {topK.map((item, index) => {
          const probPct = (item.confidence * 100).toFixed(2);
          const isWinner = index === 0;

          return (
            <div key={index} className={`topk-item ${isWinner ? 'topk-item--winner' : ''}`}>
               <div className="topk-item__info">
                  <span className="topk-item__label">{item.label}</span>
                  <span className="topk-item__value">{probPct}%</span>
               </div>
               
               <div className="topk-item__track">
                  <motion.div 
                    className={`topk-item__fill ${isWinner ? 'topk-item__fill--gold' : 'topk-item__fill--blue'}`}
                    initial={{ width: 0 }}
                    animate={{ width: `${probPct}%` }}
                    transition={{ duration: 1.2, delay: index * 0.1, ease: [0.22, 1, 0.36, 1] }}
                  />
                  {/* Subtle Glow for the top winner */}
                  {isWinner && (
                    <motion.div 
                       className="topk-item__glow"
                       animate={{ opacity: [0.3, 0.6, 0.3] }}
                       transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                       style={{ width: `${probPct}%` }}
                    />
                  )}
               </div>
            </div>
          );
        })}
      </div>

      <div className="topk-dashboard__footer">
        <p>Phân phối xác suất (Probability Distribution) giúp mô hình định lượng độ tin cậy của dự đoán.</p>
      </div>
    </div>
  );
}
