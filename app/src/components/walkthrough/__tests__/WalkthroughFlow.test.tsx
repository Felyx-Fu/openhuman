import { fireEvent, screen } from '@testing-library/react';
import { Route, Routes } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../../../test/test-utils';
import OnboardingLayout from '../../../pages/onboarding/OnboardingLayout';
import { OnboardingContext } from '../../../pages/onboarding/OnboardingContext';
import WalkthroughContainer from '../WalkthroughContainer';
import WalkthroughPhasePanel from '../WalkthroughPhasePanel';
import { WalkthroughProvider } from '../WalkthroughProvider';

const renderWalkthroughRoute = () =>
  renderWithProviders(
    <Routes>
      <Route path="/onboarding" element={<OnboardingLayout />}>
        <Route index element={<WalkthroughContainer />} />
      </Route>
    </Routes>,
    { initialEntries: ['/onboarding'] }
  );

describe('Walkthrough flow', () => {
  it('advances through setup phases and renders the review summary', () => {
    renderWalkthroughRoute();

    expect(screen.getByRole('button', { name: /complete start setup/i })).toBeInTheDocument();
    expect(screen.getByLabelText('Welcome (current)')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /complete start setup/i }));
    expect(screen.getByRole('heading', { name: 'Connect' })).toBeInTheDocument();
    expect(screen.getByLabelText('Welcome (completed)')).toBeInTheDocument();
    expect(screen.getByLabelText('Connect (current)')).toBeInTheDocument();

    for (const label of ['Gmail', 'Slack', 'WhatsApp', 'Telegram', 'Discord']) {
      fireEvent.click(screen.getByRole('button', { name: `Complete ${label}` }));
    }

    expect(screen.getByRole('heading', { name: 'Automate' })).toBeInTheDocument();

    for (const label of [
      'Daily Briefings',
      'Smart Notifications',
      'Auto Scheduling',
      'Meeting Summaries',
    ]) {
      fireEvent.click(screen.getByRole('button', { name: `Complete ${label}` }));
    }

    expect(screen.getByRole('heading', { name: 'Review' })).toBeInTheDocument();
    expect(screen.getByText('Meeting Summaries')).toBeInTheDocument();
    expect(screen.queryByText('No actions completed yet.')).not.toBeInTheDocument();
  });

  it('supports skipped and missing walkthrough states', () => {
    const skipWalkthrough = vi.fn();

    renderWithProviders(
      <OnboardingContext.Provider
        value={{
          draft: {
            connectedSources: [],
            walkthrough: {
              phase: 'connect',
              steps: [{ key: 'gmail', completed: false }],
              completed: false,
              skipped: false,
            },
          },
          setDraft: vi.fn(),
          completeAndExit: vi.fn(),
          advanceWalkthrough: vi.fn(),
          skipWalkthrough,
        }}>
        <WalkthroughContainer />
      </OnboardingContext.Provider>
    );

    fireEvent.click(screen.getByRole('button', { name: 'Skip this step' }));
    expect(skipWalkthrough).toHaveBeenCalledTimes(1);

    const { container } = renderWithProviders(
      <OnboardingContext.Provider
        value={{
          draft: { connectedSources: [] },
          setDraft: vi.fn(),
          completeAndExit: vi.fn(),
          advanceWalkthrough: vi.fn(),
          skipWalkthrough: vi.fn(),
        }}>
        <WalkthroughContainer />
      </OnboardingContext.Provider>
    );

    expect(container).toBeEmptyDOMElement();
  });

  it('renders empty review and done states', () => {
    const { rerender } = renderWithProviders(
      <WalkthroughProvider
        state={{
          phase: 'review',
          steps: [
            { key: 'gmail', completed: true },
            { key: 'slack', completed: false },
          ],
          completed: false,
          skipped: false,
        }}
        onAdvance={vi.fn()}
        onSkip={vi.fn()}>
        <WalkthroughPhasePanel />
      </WalkthroughProvider>
    );

    expect(screen.getByText('Gmail')).toBeInTheDocument();
    expect(screen.queryByText('Slack')).not.toBeInTheDocument();

    rerender(
      <WalkthroughProvider
        state={{ phase: 'review', steps: [], completed: false, skipped: true }}
        onAdvance={vi.fn()}
        onSkip={vi.fn()}>
        <WalkthroughPhasePanel />
      </WalkthroughProvider>
    );

    expect(
      screen.getByText('You skipped the setup. You can configure these anytime in Settings.')
    ).toBeInTheDocument();

    rerender(
      <WalkthroughProvider
        state={{ phase: 'done', steps: [], completed: true, skipped: false }}
        onAdvance={vi.fn()}
        onSkip={vi.fn()}>
        <WalkthroughPhasePanel />
      </WalkthroughProvider>
    );

    expect(screen.getByRole('heading', { name: "You're all set!" })).toBeInTheDocument();
  });
});
