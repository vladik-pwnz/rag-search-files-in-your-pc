const { app, BrowserWindow, ipcMain } = require('electron');
const axios = require('axios');
const path = require('path');
const { exec } = require('child_process');

function createWindow() {
    const win = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            nodeIntegration: false,
            contextIsolation: true
        }
    });

    win.loadFile('index.html');
}

app.whenReady().then(() => {
    createWindow();

    app.on('activate', function () {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

app.on('window-all-closed', function () {
    if (process.platform !== 'darwin') app.quit();
});

ipcMain.on('search-query', async (event, query) => {
    console.log(`Received search query: ${query}`);
    try {
        const response = await axios.post('http://localhost:8000/search', { query });
        console.log(`Received search results: ${response.data}`);
        event.reply('search-results', response.data);
    } catch (error) {
        console.error(`Error during search request: ${error}`);
    }
});

ipcMain.on('open-file', (event, filePath) => {
    const fullPath = path.join(__dirname, 'unparsing', filePath);
    exec(`start "" "${fullPath}"`);
});
