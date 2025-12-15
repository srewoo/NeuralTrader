/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html"
  ],
  theme: {
    extend: {
      fontFamily: {
        'heading': ['Chivo', 'sans-serif'],
        'body': ['Manrope', 'sans-serif'],
        'data': ['JetBrains Mono', 'monospace'],
      },
      colors: {
        background: '#050505',
        surface: '#0A0A0A',
        'surface-highlight': '#121212',
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        primary: {
          DEFAULT: '#3B82F6',
          foreground: 'hsl(var(--primary-foreground))'
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))'
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))'
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))'
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))'
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))'
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))'
        },
        success: '#00FF94',
        'success-dim': 'rgba(0, 255, 148, 0.1)',
        danger: '#FF2E2E',
        'danger-dim': 'rgba(255, 46, 46, 0.1)',
        'text-primary': '#EDEDED',
        'text-secondary': '#A1A1AA',
        'ai-accent': '#8B5CF6',
        foreground: 'hsl(var(--foreground))',
        chart: {
          '1': 'hsl(var(--chart-1))',
          '2': 'hsl(var(--chart-2))',
          '3': 'hsl(var(--chart-3))',
          '4': 'hsl(var(--chart-4))',
          '5': 'hsl(var(--chart-5))'
        }
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)'
      },
      boxShadow: {
        'glow-primary': '0 0 15px rgba(59, 130, 246, 0.3)',
        'glow-success': '0 0 15px rgba(0, 255, 148, 0.4)',
        'glow-danger': '0 0 15px rgba(255, 46, 46, 0.4)',
        'glow-ai': '0 0 15px rgba(139, 92, 246, 0.4)',
      },
      keyframes: {
        'accordion-down': {
          from: { height: '0' },
          to: { height: 'var(--radix-accordion-content-height)' }
        },
        'accordion-up': {
          from: { height: 'var(--radix-accordion-content-height)' },
          to: { height: '0' }
        },
        'pulse-glow': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.5' }
        },
        'typing': {
          '0%': { width: '0' },
          '100%': { width: '100%' }
        }
      },
      animation: {
        'accordion-down': 'accordion-down 0.2s ease-out',
        'accordion-up': 'accordion-up 0.2s ease-out',
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
        'typing': 'typing 2s steps(40, end)'
      }
    }
  },
  plugins: [require("tailwindcss-animate")],
}
