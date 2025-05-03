# 📁 Power BI Data Source Configuration Guide

This project uses local data sources located in the following relative path:

```
projectlens\data\output
```

To ensure the Power BI file works correctly on your system, you'll need to update the **Data Source Settings** in Power BI Desktop to reflect the correct location on your machine.

---

## ✅ Steps to Change Data Source Location in Power BI

1. **Open Power BI Desktop.**
2. Load the `.pbix` file provided with this project.
3. Navigate to the top menu:
   ```
   File > Options and settings > Data source settings
   ```
4. In the **Data source settings** window:
   - Ensure **"Data sources in current file"** is selected.
   - Select the data source path that looks similar to:
     ```
     C:\Users\YourName\OneDrive - ...\projectlens\data\output
     ```
   - Click the data source row to highlight it.
5. Click the **Change Source...** button at the bottom.
6. In the **Folder path** field:
   - Update it to the folder on your computer that follows the same sub-path:
     ```
     ...\projectlens\data\output
     ```
   - Example:
     ```
     D:\Documents\MyProjects\projectlens\data\output
     ```
7. Click **OK**, then **Close** the Data Source Settings window.
8. Refresh your data in Power BI:
   ```
   Home > Refresh
   ```

---

## ⚠️ Important Notes

- Do **not** rename the `projectlens\data\output` subfolders, or the connections may break.
- Ensure the required CSV or Excel files exist inside the updated `output` folder before refreshing.

If you encounter any issues, double-check your folder structure and try reloading the `.pbix` file.
