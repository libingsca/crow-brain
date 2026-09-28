"""Exact full-SHA256 disk set; bounded SQLite page cache, no digest truncation."""
import os,sqlite3,tempfile,atexit
from pathlib import Path
class DigestSet:
 def __init__(self):
  fd,name=tempfile.mkstemp(prefix='bi5_exact_digests_',suffix='.sqlite');os.close(fd);self.path=Path(name);self.db=sqlite3.connect(name);self.db.execute('PRAGMA journal_mode=OFF');self.db.execute('PRAGMA synchronous=OFF');self.db.execute('PRAGMA cache_size=-16384');self.db.execute('PRAGMA temp_store=FILE');self.db.execute('CREATE TABLE hashes (digest BLOB PRIMARY KEY) WITHOUT ROWID');self.count=0;self.pending=0;atexit.register(self.close)
 @staticmethod
 def key(x):
  raw=bytes.fromhex(x);assert len(raw)==32;return raw
 def __len__(self):return self.count
 def __contains__(self,x):return self.db.execute('SELECT 1 FROM hashes WHERE digest=?',(self.key(x),)).fetchone() is not None
 def update(self,items):
  before=self.db.total_changes;self.db.executemany('INSERT OR IGNORE INTO hashes VALUES (?)',((self.key(x),) for x in items));self.count+=self.db.total_changes-before;self.db.commit();self.pending=0
 def add(self,x):
  before=self.db.total_changes;self.db.execute('INSERT OR IGNORE INTO hashes VALUES (?)',(self.key(x),));self.count+=self.db.total_changes-before;self.pending+=1
  if self.pending>=1000:self.db.commit();self.pending=0
 def intersection(self,items):return {x for x in items if x in self}
 def close(self):
  if self.db is not None:self.db.close();self.db=None;self.path.unlink(missing_ok=True)
