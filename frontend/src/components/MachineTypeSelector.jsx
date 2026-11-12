import React, { useMemo } from 'react'
import {
  Box, TextField, MenuItem, Chip, Typography, ToggleButton,
  ToggleButtonGroup, Tooltip,
} from '@mui/material'
import MemoryIcon from '@mui/icons-material/Memory'
import DeveloperBoardIcon from '@mui/icons-material/DeveloperBoard'
import DnsIcon from '@mui/icons-material/Dns'

const SHAPE_CLASS_OPTIONS = [
  { value: 'any', label: 'Any Shape (Auto Best-Fit)', desc: 'Standard & High-CPU ladder (64 → 4 vCPUs)' },
  { value: 'highcpu', label: 'High-CPU (*-highcpu-*)', desc: 'Compute-dense (~2 GB RAM / vCPU)' },
  { value: 'standard', label: 'Standard (*-standard-*)', desc: 'Balanced (~4 GB RAM / vCPU)' },
  { value: 'highmem', label: 'High-Memory (*-highmem-*)', desc: 'Memory-rich (~8 GB RAM / vCPU)' },
]

/**
 * Multi-step resource selector:
 * 1. Resource Type: GPU, TPU, or CPU (toggle)
 * 2. Chip / TPU Version / CPU Series (dropdown)
 * 3. VM Configuration (exact shape) OR Shape Class (CPU vCPU Pool mode)
 */
export default function MachineTypeSelector({
  machineTypes = [],
  category, setCategory,
  chip, setChip,
  machineType, setMachineType,
  sizingMode = 'exact_shape',
  shapeClass = 'any',
  setShapeClass = null,
  disabled = false,
  size = 'small',
}) {
  const isCpuPool = category === 'CPU' && sizingMode === 'vcpu_pool'

  // Group by chip / series
  const chipGroups = useMemo(() => {
    const groups = {}
    machineTypes.forEach(mt => {
      const c = mt.chip || mt.gpu
      const cat = mt.category || 'GPU'
      if (cat !== category) return
      if (!groups[c]) {
        groups[c] = {
          chip: c,
          gpu: mt.gpu,
          processor: mt.processor || mt.gpu,
          architecture: mt.architecture || '',
          zones: mt.zones || [],
          regions: mt.regions || [],
          supported: mt.supported || {},
          types: [],
        }
      }
      groups[c].types.push(mt)
    })
    return groups
  }, [machineTypes, category])

  const chipList = useMemo(() => Object.values(chipGroups), [chipGroups])

  const selectedChipGroup = useMemo(() => chipGroups[chip] || null, [chipGroups, chip])

  const filteredMachineTypes = useMemo(() => {
    if (!selectedChipGroup) return []
    return selectedChipGroup.types
  }, [selectedChipGroup])

  const selectedMachineInfo = useMemo(() => {
    return machineTypes.find(mt => mt.machineType === machineType)
  }, [machineType, machineTypes])

  const step2Label = category === 'GPU' ? 'GPU Chip' : category === 'TPU' ? 'TPU Version' : 'CPU Series'

  return (
    <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'flex-end' }}>
      {/* Step 1: Resource Type (GPU, TPU, or CPU) */}
      <Box sx={{ pb: '20px' }}>
        <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block', fontSize: '0.7rem' }}>
          Resource Type
        </Typography>
        <ToggleButtonGroup
          value={category}
          exclusive
          onChange={(_, val) => {
            if (val) {
              setCategory(val)
              setChip('')
              setMachineType('')
            }
          }}
          size="small"
          disabled={disabled}
          sx={{ height: 40 }}
        >
          <ToggleButton value="GPU" sx={{ px: 2.2, textTransform: 'none', fontWeight: 500, gap: 0.5 }}>
            <MemoryIcon sx={{ fontSize: 16 }} /> GPU
          </ToggleButton>
          <ToggleButton value="TPU" sx={{ px: 2.2, textTransform: 'none', fontWeight: 500, gap: 0.5 }}>
            <DeveloperBoardIcon sx={{ fontSize: 16 }} /> TPU
          </ToggleButton>
          <ToggleButton value="CPU" sx={{ px: 2.2, textTransform: 'none', fontWeight: 500, gap: 0.5 }}>
            <DnsIcon sx={{ fontSize: 16 }} /> CPU
          </ToggleButton>
        </ToggleButtonGroup>
      </Box>

      {/* Step 2: Chip / Series selection */}
      <Box sx={{ minWidth: 220 }}>
        <TextField
          fullWidth
          select
          label={step2Label}
          value={chip}
          onChange={(e) => {
            setChip(e.target.value)
            setMachineType('')
          }}
          size={size}
          disabled={disabled}
          helperText={
            selectedChipGroup
              ? (category === 'CPU'
                  ? `${selectedChipGroup.processor} · ${selectedChipGroup.types.length} shapes`
                  : `${selectedChipGroup.types.length} config(s)`)
              : `Select ${category} ${category === 'CPU' ? 'series' : 'type'}`
          }
        >
          {chipList.map(cg => (
            <MenuItem key={cg.chip} value={cg.chip}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                <Chip
                  label={cg.chip}
                  size="small"
                  sx={{
                    bgcolor: category === 'CPU' ? '#e6f4ea' : '#e8f0fe',
                    color: category === 'CPU' ? '#137333' : '#1a73e8',
                    fontWeight: 600,
                    height: 22, fontSize: '0.7rem',
                  }}
                />
                <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.75rem', truncate: true }}>
                  {category === 'CPU'
                    ? `${cg.processor} (${cg.types.length})`
                    : `${cg.types.length} config${cg.types.length > 1 ? 's' : ''}`}
                </Typography>
              </Box>
            </MenuItem>
          ))}
        </TextField>
      </Box>

      {/* Step 3: Either Shape Class (for CPU vCPU Pool) OR Exact VM Configuration */}
      {isCpuPool ? (
        <Box sx={{ minWidth: 235 }}>
          <TextField
            fullWidth
            select
            label="Shape Family Preference"
            value={shapeClass}
            onChange={(e) => setShapeClass && setShapeClass(e.target.value)}
            size={size}
            disabled={disabled || !chip}
            helperText={
              SHAPE_CLASS_OPTIONS.find(o => o.value === shapeClass)?.desc || 'Auto-steps down shape ladder'
            }
          >
            {SHAPE_CLASS_OPTIONS.map(opt => (
              <MenuItem key={opt.value} value={opt.value}>
                <Box>
                  <Typography variant="body2" sx={{ fontWeight: 500, lineHeight: 1.2 }}>{opt.label}</Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>{opt.desc}</Typography>
                </Box>
              </MenuItem>
            ))}
          </TextField>
        </Box>
      ) : (
        <Box sx={{ minWidth: 240 }}>
          <TextField
            fullWidth
            select
            label="VM Configuration"
            value={machineType}
            onChange={(e) => setMachineType(e.target.value)}
            size={size}
            disabled={disabled || !chip}
            helperText={
              selectedMachineInfo
                ? (selectedMachineInfo.category === 'TPU'
                    ? `${selectedMachineInfo.gpuCount} chip(s) · ${selectedMachineInfo.vcpus || '?'} vCPUs · ${selectedMachineInfo.hbmGb || '?'} GB HBM`
                    : selectedMachineInfo.category === 'CPU'
                      ? `${selectedMachineInfo.vcpus} vCPUs · ${selectedMachineInfo.memoryGb} GB RAM · ${selectedMachineInfo.architecture}`
                      : `${selectedMachineInfo.gpu} × ${selectedMachineInfo.gpuCount}`)
                : chip ? 'Select a configuration' : `Select ${category === 'CPU' ? 'series' : 'chip'} first`
            }
          >
            {filteredMachineTypes.map(mt => (
              <MenuItem key={mt.machineType} value={mt.machineType}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center', gap: 1 }}>
                  <Box>
                    <Typography variant="body2" sx={{ fontWeight: 500, lineHeight: 1.2 }}>
                      {mt.machineType}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
                      {mt.category === 'TPU'
                        ? `${mt.gpuCount} chips · ${mt.vcpus || '?'} vCPUs · ${mt.memoryGb || '?'} GB RAM · ${mt.hbmGb || '?'} GB HBM`
                        : mt.category === 'CPU'
                          ? `${mt.family} · ${mt.vcpus} vCPUs · ${mt.memoryGb} GB RAM · ${mt.architecture}`
                          : mt.family}
                    </Typography>
                  </Box>
                  <Chip
                    label={
                      mt.category === 'TPU'
                        ? `${mt.gpuCount} chips`
                        : mt.category === 'CPU'
                          ? `${mt.vcpus} vCPUs`
                          : `${mt.gpuCount}× ${mt.chip || ''}`
                    }
                    size="small"
                    sx={{ height: 22, fontSize: '0.65rem', fontWeight: 600 }}
                  />
                </Box>
              </MenuItem>
            ))}
          </TextField>
        </Box>
      )}

      {/* Selected info chip */}
      {isCpuPool && selectedChipGroup && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, pb: '20px' }}>
          <Tooltip title={`${selectedChipGroup.processor} — ${selectedChipGroup.zones.length} zones across ${selectedChipGroup.regions.length} regions`}>
            <Chip
              icon={<DnsIcon sx={{ fontSize: 14 }} />}
              label={`${selectedChipGroup.chip} Pool | ${selectedChipGroup.zones.length} zones · ${selectedChipGroup.regions.length} regions`}
              size="small"
              variant="outlined"
              color="success"
              sx={{ height: 26, fontSize: '0.72rem', fontWeight: 500 }}
            />
          </Tooltip>
        </Box>
      )}
      {!isCpuPool && selectedMachineInfo && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, pb: '20px' }}>
          <Tooltip title={`${selectedMachineInfo.gpu} — ${selectedMachineInfo.zones?.length || 0} zones available`}>
            <Chip
              icon={selectedMachineInfo.category === 'CPU' ? <DnsIcon sx={{ fontSize: 14 }} /> : <MemoryIcon sx={{ fontSize: 14 }} />}
              label={
                selectedMachineInfo.category === 'CPU'
                  ? `${selectedMachineInfo.chip} · ${selectedMachineInfo.vcpus} vCPUs | ${selectedMachineInfo.zones?.length || 0} zones`
                  : `${selectedMachineInfo.chip} × ${selectedMachineInfo.gpuCount} | ${selectedMachineInfo.zones?.length || 0} zones`
              }
              size="small"
              variant="outlined"
              color="primary"
              sx={{ height: 26, fontSize: '0.72rem', fontWeight: 500 }}
            />
          </Tooltip>
        </Box>
      )}
    </Box>
  )
}
