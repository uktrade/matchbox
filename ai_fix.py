# AI-generated fix (fallback):
```diff
diff --git a/source.py b/source.py
index 34a5c4c..8f2a3c4 100644
--- a/source.py
+++ b/source.py
@@ -123,7 +123,7 @@
 class Source:
     # ...

     def sample(self, n: int = 100) -> pl.DataFrame:
-        return self.data.head(n)
+        if not isinstance(n, int):
+            raise ValueError("n must be an integer")
+        return self.data.head(n)

 # ...
```

PR Description: Fix `Source.sample()` to correctly handle `n` parameter. Closes #487
