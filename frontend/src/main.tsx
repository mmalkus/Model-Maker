import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { loadFeatures } from './features'
import { host } from './host'
import { LibraryPage } from './LibraryPage'

// What this workspace offers decides which controls exist, so it's read
// before the first render.
void loadFeatures().finally(() =>
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      {host.view === 'library' ? <LibraryPage /> : <App />}
    </StrictMode>,
  ),
)
