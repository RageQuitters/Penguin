# SentinelOps Dashboard

A modern machine fleet monitoring dashboard with AI-powered analysis and real-time status tracking.

## Features

- **Fleet Monitoring**: Real-time status tracking for 12 machines with color-coded health indicators
- **AI Chat Assistant**: Interactive chat interface for querying fleet status and getting recommendations
- **Comprehensive Analysis**: "Analyze All Machines" feature that generates fleet-wide summaries with engineer dispatch recommendations
- **Machine Details**: Detailed sensor readings, health metrics, and remaining useful life (RUL) tracking
- **Status Dashboard**: Quick overview of Normal, Warning, and Critical machines

## Tech Stack

- **Frontend**: React 19 + TypeScript
- **UI Components**: Radix UI + shadcn/ui
- **Styling**: Tailwind CSS 4
- **Build Tool**: Vite
- **Routing**: Wouter
- **Charts**: Recharts

## Getting Started

### Prerequisites

- Node.js 18+ 
- pnpm (recommended) or npm

### Installation

1. Extract the ZIP file
2. Install dependencies:

```bash
pnpm install
# or
npm install
```

### Development

Start the development server:

```bash
pnpm dev
# or
npm run dev
```

The application will be available at `http://localhost:3000`

### Build

Create a production build:

```bash
pnpm build
# or
npm run build
```

## Project Structure

```
sentinelops/
├── client/
│   ├── public/           # Static assets
│   ├── src/
│   │   ├── components/   # Reusable React components
│   │   ├── contexts/     # React contexts
│   │   ├── hooks/        # Custom React hooks
│   │   ├── lib/          # Utility functions and fake data
│   │   ├── pages/        # Page components
│   │   ├── App.tsx       # Main app component
│   │   ├── main.tsx      # React entry point
│   │   └── index.css     # Global styles
│   └── index.html        # HTML template
├── server/               # Backend placeholder (not used in this version)
├── shared/               # Shared types and constants
├── package.json
├── vite.config.ts
└── tsconfig.json
```

## Key Components

### Dashboard (`client/src/pages/Dashboard.tsx`)
Main application page with three-panel layout:
- Left sidebar: Machine list with status indicators
- Center: Machine details and analysis results
- Right sidebar: AI chat assistant

### AIChat (`client/src/components/AIChat.tsx`)
Interactive chat interface that responds to queries about:
- Machine status
- Critical alerts
- Recommendations
- Analysis requests

### AllMachinesAnalysis (`client/src/components/AllMachinesAnalysis.tsx`)
Comprehensive fleet analysis panel showing:
- Fleet status summary
- Engineers to dispatch
- Recommended actions
- Machine breakdown by status

### Fake Data (`client/src/lib/fakeData.ts`)
Provides:
- 12 pre-seeded machines with realistic sensor data
- Analysis functions
- AI chat response generation

## Usage

### Viewing Machine Status
1. Select a machine from the left sidebar
2. View detailed sensor readings and health metrics in the center panel
3. Check the RUL (Remaining Useful Life) progress bar

### Analyzing All Machines
1. Click the "Analyze All Machines" button in the top-right
2. Wait for the analysis to complete (simulated 1.5s delay)
3. Review the comprehensive fleet report with:
   - Status breakdown (Normal/Warning/Critical)
   - Engineers to dispatch with urgency levels
   - Recommended maintenance actions
   - Detailed machine information by status

### Chatting with AI Assistant
1. Type a question in the chat input box on the right
2. Ask about:
   - `"status"` - Get current fleet status
   - `"critical"` - See critical machines
   - `"recommend"` - Get maintenance recommendations
   - `"analyze"` - Instructions for fleet analysis
3. The AI responds with relevant information

## Customization

### Adding More Machines
Edit `client/src/lib/fakeData.ts` and add entries to the `SEED_MACHINES` array.

### Modifying Machine Data
Update the sensor values and status in the `SEED_MACHINES` array to reflect your test scenarios.

### Customizing AI Responses
Edit the `generateAIChatResponse` function in `client/src/lib/fakeData.ts` to add new patterns or responses.

### Styling
- Global styles: `client/src/index.css`
- Component styles: Use Tailwind CSS classes
- Theme colors: Defined in CSS variables in `index.css`

## Building for Production

```bash
pnpm build
```

This creates an optimized build in the `dist/` directory ready for deployment.

## License

MIT

## Support

For issues or questions, refer to the component documentation in the source files.
