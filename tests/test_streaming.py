"""Quick streaming test against live backend."""
import requests, time

url = "https://clearpath-rag-873904783482.asia-south1.run.app/query/stream"
payload = {"question": "What is the Pro plan price?", "conversation_id": "test-stream"}

r = requests.post(url, json=payload, stream=True, timeout=30)
print(f"Status: {r.status_code}")
print(f"Content-Type: {r.headers.get('content-type')}")

t0 = time.time()
count = 0
for line in r.iter_lines(decode_unicode=True):
    if line:
        count += 1
        elapsed = time.time() - t0
        preview = line[:120]
        print(f"  [{elapsed:.2f}s] chunk#{count}: {preview}")
        if count >= 15:
            print(f"  ... (truncated)")
            break

print(f"\nTotal: {time.time()-t0:.2f}s, {count} chunks received")
print("STREAMING WORKS!" if count > 2 else "NOT STREAMING - only got 1-2 chunks")
