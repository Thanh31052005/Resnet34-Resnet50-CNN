/**
 * PipelineFlow.jsx
 * ================
 * Hiển thị trực quan luồng xử lý ResNet50 từ Input → Output.
 * Mỗi layer là một node, bấm vào để xem feature maps / gradient / heatmap.
 */

import { useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './PipelineFlow.css';
import DataFlowCanvas from './DataFlowCanvas';
import CinematicFlow from './CinematicFlow';
import GlobalAvgPoolDiagram from './GlobalAvgPoolDiagram';
import KernelSweepDemo from './KernelSweepDemo';
import TopKPredictions from './TopKPredictions';
import SkipConnectionFlow from './SkipConnectionFlow';

// Cấu hình các node trong pipeline
const PIPELINE_NODES = [
  { id: 'input',   label: 'Input Image',            icon: '📷', type: 'input',      desc: 'Ảnh gốc 224×224, normalize ImageNet' },
  { id: 'conv1',   label: 'Conv 7×7',               icon: '🔲', type: 'conv',       desc: 'Tích chập đầu tiên – 64 filters, stride 2' },
  { id: 'maxpool', label: 'MaxPool 3×3',            icon: '📉', type: 'pool',       desc: 'Giảm spatial → 56×56' },
  { id: 'stage1',  label: 'Stage 1 – Edges',        icon: '🔹', type: 'stage',      desc: '3 blocks → 256 filters (edges, textures)' },
  { id: 'stage2',  label: 'Stage 2 – Patterns',     icon: '🔸', type: 'stage',      desc: '4 blocks → 512 filters (shapes)' },
  { id: 'stage3',  label: 'Stage 3 – Parts',        icon: '🔺', type: 'stage',      desc: '6 blocks → 1024 filters (object parts)' },
  { id: 'stage4',  label: 'Stage 4 – Objects',      icon: '🔳', type: 'stage',      desc: '3 blocks → 2048 filters (classes)' },
  { id: 'avgpool', label: 'Global AvgPool',         icon: '🌊', type: 'pool',       desc: 'Spatial average 7×7 → 1×1' },
  { id: 'fc',      label: 'Fully Connected',        icon: '🔗', type: 'fc',         desc: 'Vector 2048 → 1000 classes' },
  { id: 'softmax', label: 'Softmax / Prediction',   icon: '📊', type: 'fc',         desc: 'Xác suất dự đoán cuối cùng' }
];

export default function PipelineFlow({ result, activeNode, onSelectNode, viewMode }) {
  const layerMap = useMemo(() => {
    const map = {};
    if (result && result.layers) {
      result.layers.forEach(l => map[l.name] = l);
    }
    return map;
  }, [result]);

  return (
    <div className="pipeline">
      {PIPELINE_NODES.map((node, index) => {
        const isActive = activeNode === node.id;
        const layerData = layerMap[node.id];
        const hasFeatures = layerData && layerData.feature_maps && layerData.feature_maps.length > 0;

        return (
          <div key={node.id} className="pipeline-step">
            {/* Main Node Card */}
            <motion.div
              className={`pipeline-node ${isActive ? 'pipeline-node--active' : ''}`}
              onClick={() => onSelectNode(node.id)}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1, type: "spring", stiffness: 100 }}
            >
              {/* Chèn Kernel Sweep Demo ngay dưới node conv1 */}
              {node.id === 'conv1' && (
                <div style={{ marginTop: '-4px', marginBottom: '16px', zIndex: 0, position: 'relative' }}>
                   <KernelSweepDemo 
                      originalImage={result?.original_image} 
                      featureMap={layerData?.feature_maps[0]} 
                   />
                </div>
              )}

              {/* Left visual strip */}
              <div className={`pipeline-node__strip pipeline-node__strip--${node.type}`} />
              
              <div className="pipeline-node__body">
                <div className="pipeline-node__icon">{node.icon}</div>
                <div className="pipeline-node__info">
                  <div className="pipeline-node__name">{node.label}</div>
                  <div className="pipeline-node__desc">{node.desc}</div>
                </div>
                {layerData && (
                  <div className="pipeline-node__meta">
                    <span className="pipeline-node__tag">{layerData.output_shape}</span>
                  </div>
                )}
              </div>
            </motion.div>

            {/* Sub-visualizations (Conditional) */}
            <AnimatePresence>
              {isActive && (
                <motion.div 
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="pipeline-subview"
                >
                {/* GLOBAL AVG POOL DIAGRAM */}
                {node.id === 'avgpool' && (
                  <div style={{ padding: '0 18px 20px', zIndex: 0, position: 'relative' }}>
                    <GlobalAvgPoolDiagram 
                        data={layerData?.pooled_values} 
                        beforeMaps={layerMap['stage4']?.feature_maps} 
                        originalImage={result?.original_image}
                    />
                  </div>
                )}

                {/* SKIP CONNECTION FLOW (FOR STAGES) */}
                {viewMode === 'flow' && node.type === 'stage' && (
                  <motion.div initial={{opacity:0}} animate={{opacity:1}}>
                      <SkipConnectionFlow stageName={node.label} />
                  </motion.div>
                )}

                {/* FEATURE EVOLUTION GRID */}
                {viewMode === 'feature' && hasFeatures && (
                  <div className="feature-grid-wrapper">
                    <div className="feature-grid-wrapper__title">
                      🧬 Feature Evolution (Activation Focus)
                    </div>
                    <motion.div 
                      className="feature-evolution-grid"
                      variants={{
                        hidden: { opacity: 0 },
                        show: { opacity: 1, transition: { staggerChildren: 0.04 } }
                      }}
                      initial="hidden"
                      animate="show"
                    >
                      {layerData.feature_maps.map((b64, i) => (
                        <motion.div 
                          className="feature-evolution-item" 
                          key={i}
                          variants={{
                            hidden: { opacity: 0, scale: 0.9 },
                            show: { opacity: 1, scale: 1 }
                          }}
                        >
                          <div className="feature-evolution-item__img-box">
                             <img src={`data:image/png;base64,${b64}`} alt="feature" />
                             <div className="feature-evolution-item__activation-glow" />
                          </div>
                          <div className="feature-evolution-item__id">Filter #{i}</div>
                        </motion.div>
                      ))}
                    </motion.div>
                  </div>
                )}

                {/* TOP-K PREDICTIONS (ONLY FOR FC/SOFTMAX) */}
                {(node.id === 'fc' || node.id === 'softmax') && (
                  <div style={{ padding: '0 18px 20px', zIndex: 0, position: 'relative' }}>
                    {/* BẢNG DỰ ĐOÁN XÁC SUẤT TOP-5 */}
                    <TopKPredictions topK={result?.top5} />
                  </div>
                )}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Connector arrow between nodes */}
            {index < PIPELINE_NODES.length - 1 && (
              <div className="pipeline-connector">
                <div className="pipeline-connector__line" />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
