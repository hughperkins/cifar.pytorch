model_helper = {}
local L = model_helper

function L.purge_old_models(savedir, epoch, keep_epoch_mod, filename_filter)
  local model_filenames = sys.execute('cd ' .. savedir .. '; ls -t ' .. filename_filter .. ' 2>/dev/null')
  for model_filename in model_filenames:gmatch('[^\n]+') do
--    print('model_filename [' .. model_filename .. ']')
    local filename_pattern = filename_filter:gsub('*', '(%%d+)')
--    print('filename_pattern', filename_pattern)
    local this_epoch = tonumber(model_filename:gmatch(filename_pattern)())
--    print('epoch', epoch)
    if this_epoch % keep_epoch_mod ~= 0 then
      if this_epoch < epoch then
        print('purge', model_filename)
        sys.execute('rm ' .. savedir .. '/' .. model_filename)
      end
    end
  end
end

function L.get_restart_info(savedir, filename_template)
  local model_filename = sys.execute('cd ' .. savedir .. '; ls -t ' .. filename_template .. ' 2>/dev/null | head -n 1')
  if model_filename == '' then
    return
  end

  local filename_pattern = filename_template:gsub('*', '(%%d+)')
  local this_epoch = tonumber(model_filename:gmatch(filename_pattern)())
  return model_filename, this_epoch
end

return L

