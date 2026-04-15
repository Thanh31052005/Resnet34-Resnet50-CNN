const video = document.getElementById('video');
const snapshot = document.getElementById('snapshot');
const captureBtn = document.getElementById('capture-btn');
const fileUpload = document.getElementById('file-upload');
const ctx = snapshot.getContext('2d');
const journeySection = document.getElementById('journey-section');
const timelineContainer = document.getElementById('timeline-container');

navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    video.srcObject = stream;
}).catch(err => console.warn("Webcam không khả dụng", err));

captureBtn.addEventListener('click', async () => {
    snapshot.style.display = 'inline';
    ctx.drawImage(video, 0, 0, 224, 224);
    const blob = await new Promise(res => snapshot.toBlob(res, 'image/jpeg', 0.8));
    submitImage(blob, 'webcam.jpg');
});

fileUpload.addEventListener('change', async (event) => {
    if (event.target.files && event.target.files[0]) {
        const file = event.target.files[0];
        const img = new Image();
        img.onload = () => {
            snapshot.style.display = 'inline';
            ctx.drawImage(img, 0, 0, 224, 224);
        };
        img.src = URL.createObjectURL(file);
        submitImage(file, file.name);
    }
});

async function submitImage(file_blob, filename) {
    // Đã thay đổi Status Text
    document.getElementById('status-info').innerText = "⏳ Đang xử lý...";
    captureBtn.disabled = true;
    journeySection.style.display = 'none';

    const fd = new FormData();
    fd.append('file', file_blob, filename);

    try {
        const response = await fetch('/predict', { method: 'POST', body: fd });
        const data = await response.json();

        document.getElementById('status-info').innerText = "✅ Hoàn tất phân tích !";

        updateUI('r34', 't34', data.resnet34);
        updateUI('r50', 't50', data.resnet50);

        updateStageTimings('stages34', data.resnet34.stage_times);
        updateStageTimings('stages50', data.resnet50.stage_times);

        buildJourney(data.resnet34.journey, data.resnet50.journey);

    } catch (e) {
        console.error(e);
        document.getElementById('status-info').innerText = "❌ Lỗi kết nối đến Server!";
    } finally {
        captureBtn.disabled = false;
    }
}

function updateUI(resId, timeId, modelData) {
    document.getElementById(timeId).innerText = "Tổng thời gian: " + modelData.total_time;
    let html = `<div class="top1">TOP 1: ${modelData.data[0].label} (${modelData.data[0].conf})</div><hr>`;
    html += '<div class="other">';
    for (let i = 1; i < 5; i++) {
        html += `Top ${i + 1}: ${modelData.data[i].label} (${modelData.data[i].conf})<br>`;
    }
    html += '</div>';
    document.getElementById(resId).innerHTML = html;
}

function updateStageTimings(containerId, stages) {
    let html = '<table class="timing-table">';
    for (let stage in stages) {
        html += `<tr><td>${stage}:</td> <td><span>${stages[stage].toFixed(4)}s</span></td></tr>`;
    }
    html += '</table>';
    document.getElementById(containerId).innerHTML = html;
}

function buildJourney(journey34, journey50) {
    timelineContainer.innerHTML = '';

    for (let i = 0; i < journey50.length; i++) {
        let step34 = journey34[i];
        let step50 = journey50[i];

        let jsData34 = encodeURIComponent(JSON.stringify(step34.images));
        let jsData50 = encodeURIComponent(JSON.stringify(step50.images));

        let html = `
            <div class="timeline-step">
                <div class="step-title">${step50.step}</div>
                <div class="compare-container">
                    
                    <div class="compare-col" data-images="${jsData34}" data-idx="0">
                        <div class="model-name">ResNet-34 (Basic Block)</div>
                        <div class="model-desc">${step34.desc}</div>
                        <img src="${step34.images[0]}" class="large-feature-img">
                        ${renderToggleControls(step34.images.length)}
                    </div>
                    
                    <div class="compare-col" data-images="${jsData50}" data-idx="0">
                        <div class="model-name">ResNet-50 (Bottleneck)</div>
                        <div class="model-desc">${step50.desc}</div>
                        <img src="${step50.images[0]}" class="large-feature-img">
                        ${renderToggleControls(step50.images.length)}
                    </div>

                </div>
            </div>
        `;
        timelineContainer.innerHTML += html;
    }
    journeySection.style.display = 'block';
    journeySection.scrollIntoView({ behavior: 'smooth' });
}

function renderToggleControls(totalImages) {
    if (totalImages <= 1) return "";
    return `
        <div class="toggle-controls">
            <button class="toggle-btn" onclick="toggleChannel(this, -1)">◀ Prev</button>
            <span class="channel-info">Kênh 0 - 3</span>
            <button class="toggle-btn" onclick="toggleChannel(this, 1)">Next ▶</button>
        </div>
    `;
}

window.toggleChannel = function (btn, direction) {
    const colDiv = btn.closest('.compare-col');
    const imagesArray = JSON.parse(decodeURIComponent(colDiv.getAttribute('data-images')));
    let currentIdx = parseInt(colDiv.getAttribute('data-idx'));

    currentIdx += direction;
    if (currentIdx < 0) currentIdx = imagesArray.length - 1;
    if (currentIdx >= imagesArray.length) currentIdx = 0;

    colDiv.setAttribute('data-idx', currentIdx);
    colDiv.querySelector('.large-feature-img').src = imagesArray[currentIdx];

    let startChannel = currentIdx * 4;
    colDiv.querySelector('.channel-info').innerText = `Kênh ${startChannel} - ${startChannel + 3}`;
}