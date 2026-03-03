import requests, os
fname = os.path.basename(r'C:\Users\Sonakshi\OneDrive\Attachments\Desktop\PROJECTS_apps\Cyberbulling_detection_system\reports\toxicity_report_20260303_153600.html')
url = f'http://127.0.0.1:5000/download/{fname}'
print('requesting', url)
r = requests.get(url)
print('status', r.status_code)
print('content len', len(r.content))
