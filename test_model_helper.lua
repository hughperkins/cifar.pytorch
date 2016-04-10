require 'os'
require 'sys'
require 'paths'
require 'model_helper'

sys.execute('mkdir /tmp/testlogs')
for i=45,62 do
  sys.execute('touch /tmp/testlogs/model_e' .. i .. '.net')
  sys.execute('sleep 0.01')
end

model_helper.purge_old_models('/tmp/testlogs', 58, 50, 'model_e*.net')
os.execute('ls /tmp/testlogs')

latest_filename, epoch_number = model_helper.get_restart_info('/tmp/testlogs', 'model_e*.net')
print('latest_filename', latest_filename, 'epoch_number', epoch_number)

