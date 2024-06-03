const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electron', {
    search: (query) => ipcRenderer.send('search-query', query),
    onSearchResults: (callback) => ipcRenderer.on('search-results', callback),
    openFile: (filePath) => ipcRenderer.send('open-file', filePath)
});
