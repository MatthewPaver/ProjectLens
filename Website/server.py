import os
from flask import Flask, send_from_directory, make_response, render_template, jsonify, request, abort, send_file
from werkzeug.utils import secure_filename
import logging
import subprocess
import threading
import sys
import json
import time
from pathlib import Path
import csv

app = Flask(__name__)

# --- Configuration ---
# Determine absolute paths relative to this server.py file
WEBSITE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directory containing server.py
PROJECT_ROOT = os.path.abspath(os.path.join(WEBSITE_DIR, '..')) # Assumes Website/ is one level down from ProjectLens/

# Configure data directories relative to PROJECT_ROOT
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'Data', 'input')
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, 'Data', 'output')
ARCHIVE_FOLDER = os.path.join(PROJECT_ROOT, 'Data', 'archive')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.join(ARCHIVE_FOLDER, 'success'), exist_ok=True)
os.makedirs(os.path.join(ARCHIVE_FOLDER, 'failed'), exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['ARCHIVE_FOLDER'] = ARCHIVE_FOLDER

ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

# In-memory status store {project_id: {'state': str, 'msg': str}}
STATUS = {}

logging.basicConfig(level=logging.INFO)

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_project_path(project_id):
    """
    Safely construct and validate the project path
    """
    # Ensure project_id is a string and sanitise it
    project_id = str(project_id)
    if not project_id.isalnum():
        raise ValueError("Invalid project ID format")
        
    # Use UPLOAD_FOLDER instead of Projects directory
    project_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(project_id))
    
    # Ensure the final path is still under our base upload directory
    if not os.path.commonprefix([app.config['UPLOAD_FOLDER'], project_path]) == app.config['UPLOAD_FOLDER']:
        raise ValueError("Invalid project path")
        
    return project_path

def get_all_projects():
    """List directories in UPLOAD_FOLDER as projects."""
    projects = []
    try:
        for item in os.listdir(app.config['UPLOAD_FOLDER']):
            item_path = os.path.join(app.config['UPLOAD_FOLDER'], item)
            # Check if it's a directory and potentially if it contains allowed files?
            if os.path.isdir(item_path):
                 # For display, try to map ID back to a more readable name if possible
                 # This is tricky without a separate project metadata store.
                 # Simplest: use the directory name (ID) as the name too.
                 projects.append({'id': item, 'name': item}) 
    except FileNotFoundError:
        logging.warning(f"Upload folder not found: {app.config['UPLOAD_FOLDER']}")
    except Exception as e:
        logging.error(f"Error listing projects: {e}")
    return sorted(projects, key=lambda p: p['name'])

def create_new_project(name):
    """Create a directory for the project based on a sanitized name."""
    if not name or len(name.strip()) < 2:
        raise ValueError("Project name must be at least 2 characters.")
    
    # Generate a filesystem-safe ID from the name
    project_id = secure_filename(name).lower().replace('_', '-')
    if not project_id: # Handle cases where sanitization results in empty string
         raise ValueError("Invalid project name resulting in empty ID.")
         
    project_path = get_project_path(project_id)
    if os.path.exists(project_path):
        raise ValueError(f"A project with the derived ID '{project_id}' already exists.")
    try:
        os.makedirs(project_path)
        logging.info(f"Created project directory: {project_path}")
        # Return both the generated ID and the original user-provided name
        return {'id': project_id, 'name': name.strip()} 
    except Exception as e:
        logging.error(f"Error creating project directory {project_path}: {e}")
        raise

def save_to_project_folder(project_id, files):
    """Save uploaded files to the specific project folder."""
    project_path = get_project_path(project_id) # Uses secure_filename
    if not os.path.isdir(project_path):
         logging.error(f"Project folder not found for ID {project_id} at {project_path}")
         raise FileNotFoundError(f"Project '{project_id}' not found.")
    
    saved_filenames = []
    errors = []
    for file in files:
        if file and allowed_file(file.filename):
             try:
                 filename = secure_filename(file.filename) # Sanitize filename too
                 save_path = os.path.join(project_path, filename)
                 file.save(save_path)
                 saved_filenames.append(filename)
                 logging.info(f"Saved {filename} to {project_path}")
             except Exception as e:
                 logging.error(f"Error saving {filename} to {project_path}: {e}")
                 errors.append(f"Failed to save {filename}: {e}")
        elif file:
             errors.append(f"File type not allowed: {file.filename}")
             
    if errors:
        raise IOError(f"Errors during upload: {' ; '.join(errors)}")
        
    return saved_filenames

# --- Routes for HTML Pages ---
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET'])
def upload_page():
    """Renders the file upload page."""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_files():
   return jsonify({'error': 'Direct upload to /upload deprecated, use project-specific upload API.'}), 404
   
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/files')
def files_page():
    return render_template('files.html')

@app.route('/archive')
def archive_page():
    return render_template('archive.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

# --- API Endpoints ---
@app.route('/api/projects', methods=['GET'])
def list_projects():
    """Lists all projects in the input folder."""
    return jsonify(get_all_projects()), 200

@app.route('/api/archive/projects', methods=['GET'])
def list_archive_projects():
    """Lists all projects with their status (success/failed) based on archive presence."""
    projects = []
    try:
        # Check both success and failed archive folders
        success_path = os.path.join(ARCHIVE_FOLDER, 'success')
        failed_path = os.path.join(ARCHIVE_FOLDER, 'failed')

        # Only list directories (ignore files)
        success_projects = set([d for d in os.listdir(success_path) if os.path.isdir(os.path.join(success_path, d))]) if os.path.exists(success_path) else set()
        failed_projects = set([d for d in os.listdir(failed_path) if os.path.isdir(os.path.join(failed_path, d))]) if os.path.exists(failed_path) else set()
        all_projects = success_projects.union(failed_projects)

        for project in sorted(all_projects):
            status = 'success' if project in success_projects else 'failed'
            projects.append({
                'id': project,
                'name': project,
                'status': status
            })
        return jsonify(projects), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/projects', methods=['POST'])
def create_project_api():
    """Creates a new project directory."""
    logging.info(f"Received project creation request: {request.get_data(as_text=True)}")
    
    try:
        data = request.get_json()
        if not data or 'name' not in data:
            logging.error("Missing project name in request data")
            return jsonify({"error": "Missing project name"}), 400

        project_name = data['name'].strip()
        if not project_name:
            logging.error("Project name is empty after stripping")
            return jsonify({"error": "Project name cannot be empty"}), 400

        project_id = secure_filename(project_name.lower().replace(' ', '_'))
        if not project_id:
            logging.error(f"Invalid project name '{project_name}' resulting in empty ID")
            return jsonify({"error": "Invalid project name - please use only letters, numbers, spaces, or hyphens"}), 400

        logging.info(f"Project name: '{project_name}', Derived ID: '{project_id}'")

        project_path = get_project_path(project_id)
        
        if os.path.exists(project_path):
            logging.warning(f"Project creation failed: Directory '{project_path}' already exists")
            return jsonify({"error": f"A project with the ID '{project_id}' already exists"}), 400

        os.makedirs(project_path)
        logging.info(f"Created project directory: '{project_path}'")
        
        return jsonify({"id": project_id, "name": project_name}), 201
        
    except Exception as e:
        logging.error(f"Unexpected error in project creation: {str(e)}")
        return jsonify({"error": "Internal server error during project creation"}), 500

@app.route('/api/projects/<project_id>/files', methods=['GET'])
def list_project_files(project_id):
    """Lists files currently uploaded for a project."""
    try:
        project_path = get_project_path(project_id)
        if not os.path.isdir(project_path):
            return jsonify({"error": "Project not found"}), 404
        
        files = [f for f in os.listdir(project_path) 
                 if os.path.isfile(os.path.join(project_path, f)) and not f.startswith('.')]
        return jsonify(files)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"Error listing files for project '{project_id}': {e}")
        return jsonify({"error": "Failed to list project files"}), 500

@app.route('/api/projects/<project_id>/upload', methods=['POST'])
def upload_file_api(project_id):
    """
    Handle file uploads for a specific project with improved error handling
    """
    try:
        project_path = get_project_path(project_id)
        
        if not os.path.exists(project_path):
            os.makedirs(project_path)
            logging.info(f"Created project directory: {project_path}")
            
        if 'file' not in request.files:
            logging.error("No file part in request")
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            logging.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400
            
        allowed_extensions = {'csv', 'xlsx', 'xls', 'mpp', 'mpx'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            logging.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type'}), 400
            
        filename = secure_filename(file.filename)
        file_path = os.path.join(project_path, filename)
        
        if os.path.exists(file_path):
            logging.warning(f"File already exists: {file_path}")
            return jsonify({'error': 'File already exists'}), 409
            
        file.save(file_path)
        logging.info(f"Successfully saved file: {file_path}")
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename
        }), 200
        
    except ValueError as e:
        logging.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logging.error(f"Unexpected error during upload: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/projects/<project_id>/upload', methods=['POST'])
def upload_to_project_api(project_id):
    """API: Upload files into a specific project folder."""
    safe_project_id = secure_filename(project_id)
    if safe_project_id != project_id:
        logging.warning(f"Potentially unsafe project_id provided for upload: {project_id}")
        return jsonify({'error': 'Invalid project ID format.'}), 400
        
    if 'files[]' not in request.files:
        return jsonify({"error": "No file part in the request (expected 'files[]')"}), 400
        
    files = request.files.getlist('files[]')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No selected files to upload'}), 400
        
    try:
        saved = save_to_project_folder(safe_project_id, files)
        return jsonify({'message': f'Successfully uploaded {len(saved)} file(s) to {safe_project_id}.', 'filenames': saved}), 200
    except FileNotFoundError:
         return jsonify({'error': f"Project '{safe_project_id}' not found."}), 404
    except IOError as ioe:
         return jsonify({'error': str(ioe)}), 400
    except Exception as e:
        logging.error(f"API Error uploading to project {safe_project_id}: {e}")
        return jsonify({'error': 'An unexpected error occurred during upload.'}), 500

@app.route('/api/projects/<project_id>/process', methods=['POST'])
def process_project(project_id):
    """API: Trigger the processing pipeline asynchronously."""
    safe_project_id = secure_filename(project_id)
    if safe_project_id != project_id:
        logging.warning(f"Potentially unsafe project_id for processing: {project_id}")
        return jsonify(error='Invalid project ID format'), 400
        
    in_dir  = os.path.join(UPLOAD_FOLDER, safe_project_id)
    out_dir = os.path.join(OUTPUT_FOLDER, safe_project_id)
    
    if not os.path.isdir(in_dir):
        logging.error(f"Input directory not found for processing: {in_dir}")
        return jsonify(error=f'Project input data not found for {safe_project_id}'), 404
        
    os.makedirs(out_dir, exist_ok=True)
    
    current_status = STATUS.get(safe_project_id, {})
    if current_status.get('state') == 'running':
        logging.warning(f"Processing already running for project: {safe_project_id}")
        return jsonify(error='Processing is already in progress for this project.'), 409
        
    STATUS[safe_project_id] = {'state': 'running', 'msg': 'Pipeline starting…'}
    logging.info(f"Initiating processing thread for project: {safe_project_id}")

    def run_pipeline_thread():
        try:
            script_path = os.path.join(PROJECT_ROOT, 'Processing', 'main.py')
            python_executable = os.path.join(PROJECT_ROOT, '.venv', 'bin', 'python') 
            if not os.path.isfile(python_executable):
                 logging.warning(f"Virtual env python not found at {python_executable}, falling back to sys.executable ({sys.executable})")
                 python_executable = sys.executable
            
            if not os.path.isfile(script_path):
                raise FileNotFoundError(f"Processing script not found: {script_path}")

            command = [
                python_executable, 
                script_path,
                '--project_id', safe_project_id 
            ]
            
            logging.info(f"Thread executing: {' '.join(command)}")
            STATUS[safe_project_id]['msg'] = 'Running analysis. Please wait.'
            
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True
            )
            
            logging.info(f"Processing stdout for {safe_project_id}:\n{result.stdout}")
            if result.stderr:
                 logging.warning(f"Processing stderr for {safe_project_id}:\n{result.stderr}")

            STATUS[safe_project_id] = {'state': 'done', 'msg': 'Processing completed successfully!'}
            logging.info(f"Processing completed successfully for {safe_project_id}")

        except FileNotFoundError as fnf_error:
            error_msg = f"Processing script error: {fnf_error}"
            logging.error(error_msg)
            STATUS[safe_project_id] = {'state': 'error', 'msg': error_msg}
        except subprocess.CalledProcessError as proc_error:
            full_stderr = proc_error.stderr if proc_error.stderr else "(No stderr captured)"
            error_msg = f"Pipeline execution failed (Exit code {proc_error.returncode}). See logs for details."
            logging.error(f"CalledProcessError for {safe_project_id}: {error_msg}")
            logging.error(f"--- Full stderr for {safe_project_id} processing START ---")
            logging.error(full_stderr.strip())
            logging.error(f"--- Full stderr for {safe_project_id} processing END ---")
            STATUS[safe_project_id] = {'state': 'error', 'msg': f"Pipeline failed (Code {proc_error.returncode}). Check server logs."}
        except Exception as e:
            error_msg = f"An unexpected error occurred during processing: {e}"
            log_message = f"Unexpected processing error for {safe_project_id}"
            logging.exception(log_message) 
            STATUS[safe_project_id] = {'state': 'error', 'msg': error_msg}

    thread = threading.Thread(target=run_pipeline_thread, daemon=True)
    thread.start()
    
    return jsonify(message='Processing pipeline initiated.'), 202

@app.route('/api/projects/<project_id>/status', methods=['GET'])
def project_status(project_id):
    """API: Get the current processing status of a project."""
    safe_project_id = secure_filename(project_id)
    status = STATUS.get(safe_project_id, {'state': 'idle', 'msg': 'Processing not started.'})
    return jsonify(status), 200

@app.route('/api/list_outputs', methods=['GET'])
def list_outputs():
    output_files = []
    try:
        for root, dirs, files in os.walk(app.config['OUTPUT_FOLDER']):
            for filename in files:
                if not filename.startswith('.'): 
                    relative_path = os.path.relpath(os.path.join(root, filename), app.config['OUTPUT_FOLDER'])
                    output_files.append(relative_path.replace(os.path.sep, '/'))
        return jsonify(sorted(output_files))
    except Exception as e:
        logging.error(f"Error listing output files: {e}")
        return jsonify({'error': 'Could not list output files.'}), 500

@app.route('/api/download_output/<path:filepath>', methods=['GET'])
def download_output(filepath):
    logging.info(f"Attempting to download output file: {filepath}")
    if '..' in filepath or filepath.startswith('/'):
        logging.warning(f"Potential path traversal attempt blocked for: {filepath}")
        abort(400)
    check_full_path = os.path.join(app.config['OUTPUT_FOLDER'], filepath)
    if not os.path.abspath(check_full_path).startswith(os.path.abspath(app.config['OUTPUT_FOLDER'])):
        logging.error(f"Forbidden attempt to access path outside OUTPUT_FOLDER: {check_full_path}")
        abort(403)
    if not os.path.isfile(check_full_path):
        logging.warning(f"Output file not found at: {check_full_path}")
        abort(404)
    try:
        return send_from_directory(app.config['OUTPUT_FOLDER'], filepath, as_attachment=True)
    except Exception as e:
        logging.error(f"Error sending output file {filepath}: {e}", exc_info=True)
        abort(500)

@app.route('/api/preview_output/<path:filepath>', methods=['GET'])
def preview_output(filepath):
    """Return a JSON preview (header + first 20 rows) of a CSV file in Data/output/"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'output'))
    abs_path = os.path.abspath(os.path.join(base_dir, filepath))
    if not abs_path.startswith(base_dir):
        return {'error': 'Invalid file path'}, 400
    if not os.path.isfile(abs_path) or not abs_path.lower().endswith('.csv'):
        return {'error': 'File not found or not a CSV'}, 404
    try:
        with open(abs_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, [])
            rows = []
            for i, row in enumerate(reader):
                if i >= 20:
                    break
                rows.append(row)
        return {'header': header, 'rows': rows}
    except Exception as e:
        return {'error': f'Failed to read file: {str(e)}'}, 500

@app.route('/api/list_archive', methods=['GET'])
def list_archive():
    archived_files = []
    try:
        for status_folder in ['success', 'failed']:
            current_dir = os.path.join(app.config['ARCHIVE_FOLDER'], status_folder)
            if os.path.exists(current_dir):
                 for project_name in os.listdir(current_dir):
                      project_path = os.path.join(current_dir, project_name)
                      if os.path.isdir(project_path):
                           for filename in os.listdir(project_path):
                                if not filename.startswith('.') and os.path.isfile(os.path.join(project_path, filename)):
                                     relative_path = os.path.join(status_folder, project_name, filename)
                                     archived_files.append(relative_path.replace(os.path.sep, '/'))
        return jsonify(sorted(archived_files))
    except Exception as e:
        logging.error(f"Error listing archive files: {e}")
        return jsonify({'error': 'Could not list archive files.'}), 500

@app.route('/api/download_archive/<path:filepath>', methods=['GET'])
def download_archive(filepath):
    logging.info(f"Attempting to download archive file: {filepath}")
    if '..' in filepath or filepath.startswith('/'):
        logging.warning(f"Potential path traversal attempt blocked for archive: {filepath}")
        abort(400)
    check_full_path = os.path.join(app.config['ARCHIVE_FOLDER'], filepath)
    if not os.path.abspath(check_full_path).startswith(os.path.abspath(app.config['ARCHIVE_FOLDER'])):
        logging.error(f"Forbidden attempt to access path outside ARCHIVE_FOLDER: {check_full_path}")
        abort(403)
    if not os.path.isfile(check_full_path):
        logging.warning(f"Archive file not found at: {check_full_path}")
        abort(404)
    try:
        return send_from_directory(app.config['ARCHIVE_FOLDER'], filepath, as_attachment=True)
    except Exception as e:
        logging.error(f"Error sending archive file {filepath}: {e}", exc_info=True)
        abort(500)

@app.route('/api/projects/<project_id>/archive', methods=['GET'])
def list_project_archive(project_id):
    """Lists archive files for a specific project."""
    try:
        success_path = os.path.join(ARCHIVE_FOLDER, 'success', project_id)
        failed_path = os.path.join(ARCHIVE_FOLDER, 'failed', project_id)
        
        if os.path.isdir(success_path):
            path = success_path
        elif os.path.isdir(failed_path):
            path = failed_path
        else:
            return jsonify([]), 200
            
        files = sorted([f for f in os.listdir(path) 
                       if os.path.isfile(os.path.join(path, f)) and not f.startswith('.')])
        return jsonify(files), 200
        
    except Exception as e:
        logging.error(f"Error listing archive for project {project_id}: {e}")
        return jsonify({"error": f"Failed to list archive files"}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    response = make_response(send_from_directory('static', filename))
    response.headers['Cache-Control'] = 'public, max-age=31536000'
    return response

if __name__ == '__main__':
    app.run(debug=True)