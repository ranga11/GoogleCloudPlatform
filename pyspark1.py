#

import sys
import imp

print(sys.executable, sys.version_info)
for package in sys.argv[1:]:
  print(imp.find_module(package))

# COMMAND ----------

#! pip install findspark

# COMMAND ----------


