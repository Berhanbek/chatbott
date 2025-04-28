"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Search } from "lucide-react"
import { Input } from "@/components/ui/input"
import { useState } from "react"

type FacultyMember = {
  name: string
  email: string
  office: string
}

export function FacultyList() {
  const [searchQuery, setSearchQuery] = useState("")

  const facultyMembers: FacultyMember[] = [
    { name: "W/ro Adey Edessa", email: "adey.edessa@aau.edu.et", office: "Room 113" },
    { name: "W/t Amina Abdulkadir", email: "amina.abdulkadir@aau.edu.et", office: "Room 122" },
    { name: "Ato Aminu Mohammed", email: "aminu.mohammed@aau.edu.et", office: "Room 319" },
    { name: "Ato Andargachew Asfaw", email: "andargachew.asfaw@aau.edu.et", office: "Room 424" },
    { name: "W/t Dagmawit Mohammed", email: "dagmawit.mohammed@aau.edu.et", office: "Room 122" },
    { name: "Dr. Dereje Teferi", email: "dereje.teferi@aau.edu.et", office: "Room 419" },
    { name: "Dr. Ermias Abebe", email: "ermias.abebe@aau.edu.et", office: "Room 115" },
    { name: "Dr. Getachew H/Mariam", email: "getachew.hailemariam@aau.edu.et", office: "Room 618" },
    { name: "Ato Gebremichael Meshesha", email: "gebremichael.meshesha@aau.edu.et", office: "Room 122" },
    { name: "Ato Kidus Menfes", email: "kidus.menfese@aau.edu.et", office: "Room 511" },
    { name: "W/o Lemlem Hagos", email: "lemlem.hagos@aau.edu.et", office: "Room 116" },
    { name: "Dr. Lemma Lessa", email: "lemma.lessa@aau.edu.et", office: "Room 417" },
    { name: "Dr. Martha Yifiru", email: "martha.yifiru@aau.edu.et", office: "Room 420" },
    { name: "Ato Melaku Girma", email: "melaku.girma@aau.edu.et", office: "Room 224" },
    { name: "W/o Meseret Hailu", email: "meseret.hailu@aau.edu.et", office: "Room 113" },
    { name: "Dr. Melekamu Beyene", email: "melekamu.beyene@aau.edu.et", office: "Room 423" },
    { name: "Ato Miftah Hassen", email: "miftah.hassen@aau.edu.et", office: "Room 424" },
    { name: "W/t Mihiret Tibebe", email: "mihret.tibebe@aau.edu.et", office: "Room 113" },
    { name: "Dr. Million Meshesha", email: "million.meshesha@aau.edu.et", office: "Room 418" },
    { name: "Dr. Rahel Bekele", email: "rahel.bekele@aau.edu.et", office: "Room 221" },
    { name: "Dr. Solomon Tefera", email: "solomon.teferra@aau.edu.et", office: "Room 421" },
    { name: "Dr. Temtem Assefa", email: "temtem.assefa@aau.edu.et", office: "Room 622" },
    { name: "Ato Teshome Alemu", email: "teshome.alemu@aau.edu.et", office: "Room 224" },
    { name: "Ato Tsegaye Berhanu", email: "tsegaye.berhanu@aau.edu.et", office: "Room 319" },
    { name: "Ato Wendwesen Endale", email: "wendwesen.endale@aau.edu.et", office: "Room 319" },
    { name: "Dr. Wondwossen Mulugeta", email: "wondwossen.mulugeta@aau.edu.et", office: "Room 114" },
    { name: "Dr. Workshet Lamenew", email: "workshet.lamenew@aau.edu.et", office: "Room 222" },
  ]

  const filteredFaculty = facultyMembers.filter(
    (member) =>
      member.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      member.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
      member.office.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  return (
    <Card className="retro-border bg-secondary/50">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm terminal-text">Faculty & Staff Directory</CardTitle>
      </CardHeader>
      <CardContent className="text-xs space-y-4 terminal-text">
        <div className="relative">
          <Search className="absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search faculty..."
            className="pl-8 h-7 text-xs"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        <ScrollArea className="h-[300px] pr-4">
          <div className="space-y-3">
            {filteredFaculty.map((member, index) => (
              <div key={index} className="p-2 rounded-md bg-background/50 retro-border">
                <p className="font-bold text-primary">{member.name}</p>
                <p className="mt-1">{member.email}</p>
                <p>{member.office}</p>
              </div>
            ))}
            {filteredFaculty.length === 0 && (
              <div className="text-center py-4 text-muted-foreground">
                No faculty members found matching your search.
              </div>
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
