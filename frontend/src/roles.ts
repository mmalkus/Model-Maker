import { useState } from 'react'

// Mirrors modelmaker.packet.ColumnRole -- keep in sync with that enum.
export const ROLE_LABELS: Record<string, string> = {
  id: 'ID',
  target: 'Target',
  predicted: 'Predicted',
  weight: 'Weight',
  feature: 'Feature',
  date: 'Date',
  segment: 'Segment',
  excluded: 'Excluded',
  unassigned: '(clear tag)',
}

// Roles a user can hand-pick for a column. "predicted" is deliberately
// excluded -- it's applied automatically by modelling blocks (see
// blocks/modelling.py), never something to tag on a raw input column.
export const ASSIGNABLE_ROLES = ['id', 'target', 'weight', 'date', 'feature', 'segment', 'excluded'] as const

// Whether "AI analyze data" should also set roles on columns that have none
// yet -- one remembered preference shared by every analyze button.
const ASSIGN_ROLES_KEY = 'modelmaker.analyze.assignRoles'

export function useAssignRolesPref(): [boolean, (v: boolean) => void] {
  const [value, setValue] = useState(() => {
    try {
      return localStorage.getItem(ASSIGN_ROLES_KEY) !== 'false'
    } catch {
      return true
    }
  })
  const set = (v: boolean) => {
    setValue(v)
    try {
      localStorage.setItem(ASSIGN_ROLES_KEY, String(v))
    } catch {
      // storage unavailable -- the choice just won't be remembered
    }
  }
  return [value, set]
}
