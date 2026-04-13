/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Brand palette from the design system
        lime: {
          DEFAULT: "#D5FF40",
          50: "#F7FFD9",
          100: "#EEFFB3",
          200: "#E3FF7D",
          300: "#D5FF40",
          400: "#BCE62C",
          500: "#9FC91A",
          600: "#7FA112",
          700: "#5B7409",
          800: "#3D4D05",
          900: "#232B02",
        },
        bone: {
          DEFAULT: "#C0C2B8",
          50: "#F4F5F2",
          100: "#E9EAE5",
          200: "#D6D8D0",
          300: "#C0C2B8",
          400: "#A2A59A",
          500: "#83877B",
          600: "#686C61",
          700: "#4E5149",
          800: "#35372F",
          900: "#1D1E19",
        },
        ink: {
          // Cool slate surfaces to make the app feel more like a control plane
          DEFAULT: "#0F172A",
          50: "#314866",
          100: "#2A3E59",
          200: "#22344B",
          300: "#1B2B3F",
          400: "#162334",
          500: "#0F172A",
          600: "#0B1220",
          700: "#080E19",
          800: "#050912",
          900: "#02050B",
        },
      },
      fontFamily: {
        sans: [
          "Fira Sans",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        display: ["Fira Sans", "system-ui", "sans-serif"],
        mono: [
          "Fira Code",
          "JetBrains Mono",
          "SFMono-Regular",
          "Menlo",
          "monospace",
        ],
      },
      boxShadow: {
        "glow-lime": "0 0 0 1px rgba(213, 255, 64, 0.35), 0 10px 30px -12px rgba(213, 255, 64, 0.35)",
        "panel": "0 1px 0 rgba(255,255,255,0.04) inset, 0 18px 40px -24px rgba(0,0,0,0.6)",
      },
      borderRadius: {
        xl: "1rem",
        "2xl": "1.5rem",
      },
      backgroundImage: {
        "grid-ink": "radial-gradient(rgba(255,255,255,0.04) 1px, transparent 1px)",
      },
    },
  },
  plugins: [],
};
