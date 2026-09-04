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
