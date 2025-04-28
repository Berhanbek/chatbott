import { Card, CardContent } from "@/components/ui/card"
import { PageHeader } from "@/components/page-header"
import { MapPin, Phone, Mail, BookOpen, Users, Calendar } from "lucide-react"

export default function DepartmentInfo() {
  return (
    <div className="container py-8 px-4 md:px-6">
      <PageHeader
        title="Welcome to the School of Information Science"
        description="Learn about our department, mission, and facilities"
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-8">
        <div className="md:col-span-2 space-y-6">
          <Card className="overflow-hidden">
            <div className="relative h-48 md:h-64">
              <div className="absolute inset-0 bg-gradient-to-r from-primary/20 to-secondary/20 z-10"></div>
              <div className="absolute inset-0 hero-pattern"></div>
              <div className="absolute bottom-4 left-4 z-20">
                <h3 className="text-2xl font-bold text-white">School of Information Science</h3>
                <p className="text-white/80">College of Natural and Computational Sciences</p>
              </div>
            </div>
            <CardContent className="p-6">
              <h3 className="text-xl font-semibold mb-4">Department Overview</h3>
              <p className="mb-4">
                The School of Information Science is a vital part of the College of Natural and Computational Sciences
                at Addis Ababa University. It is dedicated to empowering students with both technical expertise and
                managerial insight across diverse areas in Information Systems.
              </p>
              <p>
                Our department combines rigorous academic training with practical skills development, preparing
                graduates who can address complex information challenges in various sectors of the Ethiopian economy and
                beyond.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <h3 className="text-xl font-semibold mb-4">Our Mission</h3>
              <p className="mb-4">The mission of the School of Information Science is to:</p>
              <ul className="space-y-2 list-disc pl-5">
                <li>Provide high-quality education in information systems and related fields</li>
                <li>Conduct innovative research that addresses local and global challenges</li>
                <li>Develop professionals who can lead digital transformation initiatives</li>
                <li>Foster collaboration between academia, industry, and government</li>
                <li>Contribute to the advancement of information science in Ethiopia and Africa</li>
              </ul>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <h3 className="text-xl font-semibold mb-4">Academic Goals</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-muted rounded-lg">
                  <h4 className="font-medium text-primary mb-2">Excellence in Teaching</h4>
                  <p className="text-sm">Deliver cutting-edge curriculum that balances theory and practice</p>
                </div>
                <div className="p-4 bg-muted rounded-lg">
                  <h4 className="font-medium text-primary mb-2">Research Impact</h4>
                  <p className="text-sm">Conduct research that addresses real-world information challenges</p>
                </div>
                <div className="p-4 bg-muted rounded-lg">
                  <h4 className="font-medium text-primary mb-2">Industry Engagement</h4>
                  <p className="text-sm">Maintain strong connections with IT industry partners</p>
                </div>
                <div className="p-4 bg-muted rounded-lg">
                  <h4 className="font-medium text-primary mb-2">Community Service</h4>
                  <p className="text-sm">Contribute to digital literacy and technology adoption</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6">
          <Card className="card-hover">
            <CardContent className="p-6">
              <h3 className="text-xl font-semibold mb-4">Contact Information</h3>
              <div className="space-y-4">
                <div className="flex items-start">
                  <MapPin className="h-5 w-5 text-primary mr-3 mt-0.5" />
                  <div>
                    <p className="font-medium">Location</p>
                    <p className="text-muted-foreground text-sm">Eshetu Chole Building, FBE Campus, 1st–6th Floors</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <Phone className="h-5 w-5 text-primary mr-3 mt-0.5" />
                  <div>
                    <p className="font-medium">Phone</p>
                    <p className="text-muted-foreground text-sm">+251 11 122 9191</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <Mail className="h-5 w-5 text-primary mr-3 mt-0.5" />
                  <div>
                    <p className="font-medium">Email</p>
                    <p className="text-muted-foreground text-sm">info@aau.edu.et</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="card-hover">
            <CardContent className="p-6">
              <h3 className="text-xl font-semibold mb-4">Department Head</h3>
              <div className="flex items-center mb-4">
                <div className="h-16 w-16 rounded-full bg-muted flex items-center justify-center mr-4">
                  <Users className="h-8 w-8 text-primary" />
                </div>
                <div>
                  <p className="font-medium">Dr. Michael Melese</p>
                  <p className="text-muted-foreground text-sm">michael.melese@aau.edu.et</p>
                  <p className="text-muted-foreground text-sm">Office: Eshetu Chole 621</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="card-hover">
            <CardContent className="p-6">
              <h3 className="text-xl font-semibold mb-4">Student Resources</h3>
              <div className="space-y-3">
                <div className="flex items-start">
                  <BookOpen className="h-5 w-5 text-primary mr-3 mt-0.5" />
                  <div>
                    <p className="font-medium">Resource Bot</p>
                    <p className="text-muted-foreground text-sm">
                      Access lecture notes, PDFs, past exams via @SISResourcesBot on Telegram
                    </p>
                  </div>
                </div>
                <div className="flex items-start">
                  <Calendar className="h-5 w-5 text-primary mr-3 mt-0.5" />
                  <div>
                    <p className="font-medium">Events & Activities</p>
                    <p className="text-muted-foreground text-sm">Hackathons, IS Talks, Game Fests, and more</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="relative h-48 rounded-lg overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-t from-background to-transparent z-10"></div>
            <div className="absolute inset-0 hero-pattern"></div>
            <div className="absolute bottom-4 left-4 z-20">
              <p className="text-white/80 text-sm">Powered by</p>
              <h3 className="text-xl font-bold text-white">SiS Super Bot</h3>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
