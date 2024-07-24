from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

gauth = GoogleAuth()
drive = GoogleDrive(gauth)

def upload_file_list(upload_file_list):
    for upload_file in upload_file_list: 
        gfile = drive.CreateFile({'parents': [{'id': '1pzschX3uMbxU0lB5WZ6IlEEeAUE8MZ-t'}]})
        gfile.SetContentFile(upload_file) 
        gfile.Upload()

def list_out_files():
    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format('1cIMiqUDUNldxO6Nl-KVuS9SV-cWi9WLi')}).GetList()
    for file in file_list: 
        print('title: %s, id: %s' % (file['title'], file['id']))

def download_files(file_list):
    for i, file in enumerate(sorted(file_list, key = lambda x: x['title']), start=1): 
        print('Downloading {} file from GDrive ({}/{})'.format(file['title'], i, len(file_list))) 
        file.GetContentFile(file['title'])