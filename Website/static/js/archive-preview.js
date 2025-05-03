window.addEventListener('previewFile', async function(e) {
    const filepath = e.detail;
    const previewModal = document.getElementById('previewModal');
    const previewFileName = document.getElementById('previewFileName');
    const previewTableHead = document.getElementById('previewTableHead');
    const previewTableBody = document.getElementById('previewTableBody');
    if (!filepath || !previewModal) return;
    // Show filename with ellipsis and tooltip
    const shortName = filepath.split('/').pop();
    previewFileName.innerHTML = `Preview: <span style="max-width: 300px; display: inline-block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; vertical-align: bottom;" title="${shortName}">${shortName}</span>`;
    previewTableHead.innerHTML = '';
    previewTableBody.innerHTML = '<tr><td colspan="10" class="text-center py-4"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div></td></tr>';
    try {
        const response = await fetch(`/api/download_archive/${encodeURIComponent(filepath)}`);
        if (!response.ok) throw new Error('Failed to fetch file');
        const text = await response.text();
        // Parse CSV (robust, handles quoted fields)
        function parseCSVRows(str, maxRows) {
            const rows = [];
            let row = [], field = '', inQuotes = false, i = 0, c;
            for (let line of str.split(/\r?\n/)) {
                if (!line.trim()) continue;
                row = [];
                field = '';
                inQuotes = false;
                for (i = 0; i < line.length; i++) {
                    c = line[i];
                    if (c === '"') {
                        if (inQuotes && line[i+1] === '"') { field += '"'; i++; }
                        else inQuotes = !inQuotes;
                    } else if (c === ',' && !inQuotes) {
                        row.push(field); field = '';
                    } else {
                        field += c;
                    }
                }
                row.push(field);
                rows.push(row);
                if (maxRows && rows.length >= maxRows) break;
            }
            return rows;
        }
        const rows = parseCSVRows(text, 11); // header + 10 rows
        if (!rows.length) {
            previewTableHead.innerHTML = '';
            previewTableBody.innerHTML = '<tr><td class="text-center">No data</td></tr>';
        } else {
            const headers = rows[0];
            previewTableHead.innerHTML = '<tr>' + headers.map(h => `<th style='white-space:nowrap;'>${h}</th>`).join('') + '</tr>';
            previewTableBody.innerHTML = rows.slice(1).map(row => {
                return '<tr>' + headers.map((_, i) => `<td style='max-width:180px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;' title='${row[i]||''}'>${row[i]||''}</td>`).join('') + '</tr>';
            }).join('');
        }
        // Show modal after content is loaded
        const modal = bootstrap.Modal.getOrCreateInstance(previewModal);
        modal.show();
    } catch (err) {
        previewTableHead.innerHTML = '';
        previewTableBody.innerHTML = `<tr><td class='text-danger'>Error loading preview</td></tr>`;
        const modal = bootstrap.Modal.getOrCreateInstance(previewModal);
        modal.show();
    }
});
