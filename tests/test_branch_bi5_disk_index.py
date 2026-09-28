import hashlib
from branch_bi5_disk_index import DigestSet

def test_full_digest_set_matches_in_memory_operations():
 xs=[hashlib.sha256(str(i).encode()).hexdigest() for i in range(3000)];a=DigestSet();b=set()
 try:
  a.update(xs[:1200]+xs[:30]);b.update(xs[:1200]+xs[:30]);assert len(a)==len(b)
  for x in xs[900:2200]:
   assert (x in a)==(x in b);a.add(x);b.add(x)
  assert len(a)==len(b);assert a.intersection(xs[2000:])==b.intersection(xs[2000:])
  a.update(iter(xs));b.update(xs);assert len(a)==len(b)==3000
  for x in xs:assert x in a
  assert a.db.execute('SELECT min(length(digest)),max(length(digest)) FROM hashes').fetchone()==(32,32)
 finally:
  path=a.path;a.close();assert not path.exists()
