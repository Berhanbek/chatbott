import { Button } from "@/components/ui/button"
import Link from "next/link"

export function HeroSection() {
  return (
    <section className="relative py-20 md:py-28 overflow-hidden">
      <div className="absolute inset-0 hero-pattern"></div>
      <div className="absolute inset-0 bg-gradient-to-r from-background/90 to-background/80"></div>

      <div className="container relative z-10 px-4 md:px-6">
        <div className="max-w-3xl mx-auto text-center">
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6">
            Welcome to <span className="gold-gradient">SiS Super Bot</span>
          </h1>
          <p className="text-xl md:text-2xl text-muted-foreground mb-8">
            Your intelligent assistant for navigating academic life at the School of Information Science
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/department-info">
              <Button size="lg" className="bg-primary hover:bg-primary/90 w-full sm:w-auto">
                Explore Department
              </Button>
            </Link>
            <Link href="/gpa-calculator">
              <Button size="lg" variant="outline" className="w-full sm:w-auto">
                Calculate GPA
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </section>
  )
}
