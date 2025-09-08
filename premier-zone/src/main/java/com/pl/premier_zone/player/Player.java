package com.pl.premier_zone.player;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.*;

@Entity
@Table(name = "player_stats", schema = "public")
@Getter @Setter @NoArgsConstructor @AllArgsConstructor
public class Player {

    @Id
    @Column(name = "player_name", unique = true)
    private String name;

    @Column(name = "nation")
    private String nation;

    @Column(name = "position")
    private String pos;

    @Column(name = "age")
    private Integer age;

    @Column(name = "matches_played")
    private Integer mp;

    @Column(name = "starts")
    private Integer starts;

    @Column(name = "minutes_played")
    private Integer min;

    @Column(name = "goals")
    private Integer gls;

    @Column(name = "assists")
    private Integer ast;

    @Column(name = "penalties_scored")
    private Integer pk;

    @Column(name = "yellow_cards")
    private Integer crdy;

    @Column(name = "red_cards")
    private Integer crdr;

    @Column(name = "expected_goals")
    private Double xg;

    @Column(name = "expected_assists")
    private Double xag;

    @Column(name = "team_name")
    private String team;
}
