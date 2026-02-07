"""Check PDF access."""
import requests

url = "http://repository.its.ac.id/111537/1/5025201104-Undergraduate_Thesis.pdf"
r = requests.get(url, allow_redirects=True, timeout=15)
print(f"Status: {r.status_code}")
print(f"Final URL: {r.url}")
print(f"Content-Type: {r.headers.get('content-type', 'unknown')}")
print(f"Content-Length: {r.headers.get('content-length', 'unknown')}")

# Check if it's HTML (login page redirect)
if r.status_code == 200 and 'html' in r.headers.get('content-type', ''):
    print("\nGot HTML instead of PDF - checking content:")
    print(r.text[:500])
