/**
 * electron/main.js
 * ================
 * Electron main process – tạo cửa sổ BrowserWindow và load Vite dev server
 * (khi dev) hoặc build sản phẩm (khi production).
 */

const { app, BrowserWindow } = require('electron');
const path = require('path');

const isDev = !app.isPackaged;

function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    title: 'ResNet50 Visualizer',
    backgroundColor: '#0a0e1a',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
    // Ẩn menu bar mặc định
    autoHideMenuBar: true,
  });

  if (isDev) {
    // Khi dev: load từ Vite dev server
    win.loadURL('http://localhost:5173');
    // Mở DevTools tự động khi dev
    win.webContents.openDevTools({ mode: 'detach' });
  } else {
    // Khi build: load từ file tĩnh
    win.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
  }
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
